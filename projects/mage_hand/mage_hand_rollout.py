










import numpy as np
from diffusion_policy.rollout import Rollout

# import pytorch utils
from diffusion_policy.common import pytorch_util
from diffusion_policy import utils
            
class MageHandRollout(Rollout):
    def convert_obs(self, obs):
        """
        I'm not overly concerned about rel_fk btw, so I'll leave it for now
        """
        # extract pose_pitch_roll from the pose_state
        pose_state = obs['pose_state']
        
        pose_pitch_roll = utils.drake_extract_pitch_roll_from_pose(pose_state)
        
        # get yaw only pose state
        pose_yaw_only = utils.drake_compute_yaw_only_pose_from_pose(pose_state)
        
        # save for use in convert_action
        self.current_pose_yaw_only = pose_yaw_only
        
        # get obs poses
        object_pose = obs['object_pose']
        object_target_pose = obs['object_target_pose']
        
        # compute rel positions based on yaw only pose
        rel_object_pos = utils.drake_compute_rel_pos(pose_yaw_only, object_pose)
        rel_object_target_pos = utils.drake_compute_rel_pos(pose_yaw_only, object_target_pose)
        
        
        obs['pose_pitch_roll'] = pose_pitch_roll
        obs['rel_object_pos'] = rel_object_pos
        obs['rel_object_target_pos'] = rel_object_target_pos
        
        return obs
    
    def save_data(self, obs):
        # convert
        obs = self.convert_obs(obs)
        
        # save to evaluator
        self.evaluator.save_data(obs)
    
    def convert_action(self, action_dict):
        """
        joint action requires no change
        rel_pos_action and rel_quat_action need to be converted to pos_action and quat_action w.r.t. the saved current_pose_yaw_only (because that's how the actions were relativized)
        
        drake is expecting an pose action
        """
        assert(self.current_pose_yaw_only is not None)
        
        # extract rel pos and rel rpy
        rel_pos_action = action_dict['rel_pos_action']
        rel_quat_action = action_dict['rel_quat_action']
        
        # normalize the rel quat action to make it valid
        utils.normalize_quaternions_inplace(rel_quat_action)
        
        # concat quat and pos into a pose
        rel_pose_action = np.concatenate([rel_quat_action, rel_pos_action], axis=-1)
        
        # compute abs pose from rel pose
        pose_action = utils.drake_compute_abs_pose_from_rel(
            src_pose = self.current_pose_yaw_only,
            rel_pose = rel_pose_action
        )
        
        action_dict['pose_action'] = pose_action
        
        return action_dict
    
    def rollout_prep(self):
        # update the eval class
        
        # reset the sim class
        obs, info = self.env.reset()
        
        # resetting once doesn't reset everything, so as a hack we can just reset again
        obs, info = self.env.reset()
        
        obs = self.convert_obs(obs)
                        
        # update the evaluator
        self.save_data(obs)
                
    def one_rollout(self):
        """
        Run one rollout
        """
        samples = []
        total_reward = 0.0
        best_qvals = []
        total_jerk = 0.0
        
        done = False
        while not done:
            # get the action trajectory
            actions, best_qval, failed = self.infer_action() 
            
            if failed:
                print("No valid action. Episode failure.")
                done = True
                break
            
            # none protection
            if actions is not None:
                actions = self.prep_action_for_stepping(actions)
                
                best_qvals.append(best_qval)
                
                # execute the full trajectory
                new_samples, new_obs, rewards, dones, infos, jerk = self.step_trajectory(actions)
                
                # append all new samples
                samples += new_samples
                
                # update the evaluator
                self.save_data(
                    new_obs,
                )
                
                if dones:
                    done = True
                    
                # logging
                total_reward += rewards
                total_jerk += jerk
                
        # normalize jerk by episode length
        if len(samples) > 0:
            total_jerk /= len(samples)
            
        return samples, total_reward, best_qvals, total_jerk

# immediate next steps:
    # collect larger mage hand demo dataset
        
    
    def step_trajectory(self, actions):
        total_reward = 0.0
        samples = []
        
        new_obs = None
        done = False
        infos = None
        
        
        total_jerk = 0.0
        
        m = min(len(actions), self.rollout_num_actions)
        for i in range(m):
            action = actions[i]
            
            # convert action for Drake
            action = self.convert_action(action)
            
            # step the env
            new_obs, rewards, terminated, truncated, infos = self.env.step(action) #type:ignore
                        
            done = terminated or truncated
            
            total_reward += float(rewards)
            
            if done:
                break
            
        return samples, new_obs, total_reward, done, infos, total_jerk
    
    def prep_action_for_stepping(self, actions):
        """
        for mage hand, we need to convert from a dict of traj to a list of dicts
        """
        assert(isinstance(actions, dict))
        
        # dict apply x.numpy()
        actions_np = pytorch_util.dict_to_numpy(actions)
        
        ## zip each waypoint in the action trajectory into a list of actions
        
        # assumes the first dim is the batch dim and the second dim is the traj dim
        assert(len(actions_np[list(actions_np.keys())[0]].shape) == 3)
        num_waypoints = actions_np[list(actions_np.keys())[0]].shape[1]
        
        # Reconstruct into a list of dictionaries
        waypoint_actions = [
            {key: val[:, i, :] for key, val in actions_np.items()}
            for i in range(num_waypoints)
        ]
        
        return waypoint_actions