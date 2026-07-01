# want to add another obs to the policy?
- add to obs_keys_to_load
- add to obs_keys_to_use
- add to sandbox obs
- add to sandbox gym spec

# example json config for debugging & running:
"""
{
    "name": "train aiet-erlenmeyer-flask",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/train.py",
    "console": "integratedTerminal",
    "args": [
        "--config-dir=./diffusion_policy/config/dexnex",
        "--config-name=aiet_erlenmeyer_flask_1.yaml"
    ]
},
"""