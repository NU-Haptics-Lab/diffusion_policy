

from diffusion_policy.rollout import Rollout

# import pytorch utils
from diffusion_policy.common import pytorch_util
            
class MageHandRollout(Rollout):
    def convert_obs(self, obs):
        return obs
    
    def rollout_prep(self):
        # update the eval class
        
        # reset the sim class
        obs, info = self.env.reset()
        
        # resetting once doesn't reset everything, so as a hack we can just reset again
        obs, info = self.env.reset()
                        
        # update the evaluator
        self.evaluator.save_data(obs)
                
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
                self.evaluator.save_data(
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