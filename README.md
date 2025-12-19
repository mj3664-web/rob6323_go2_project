# ROB6323 Go2 Project — Isaac Lab

This repository is the starter code for the NYU Reinforcement Learning and Optimal Control project in which students train a Unitree Go2 walking policy in Isaac Lab starting from a minimal baseline and improve it via reward shaping and robustness strategies. Please read this README fully before starting and follow the exact workflow and naming rules below to ensure your runs integrate correctly with the cluster scripts and grading pipeline.

## Repository policy

- Fork this repository and do not change the repository name in your fork.  
- Your fork must be named rob6323_go2_project so cluster scripts and paths work without modification.

### Prerequisites

- **GitHub Account:** You must have a GitHub account to fork this repository and manage your code. If you do not have one, [sign up here](https://github.com/join).

### Links
1.  **Project Webpage:** [https://machines-in-motion.github.io/RL_class_go2_project/](https://machines-in-motion.github.io/RL_class_go2_project/)
2.  **Project Tutorial:** [https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md](https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md)

## Connect to Greene

- Connect to the NYU Greene HPC via SSH; if you are off-campus or not on NYU Wi‑Fi, you must connect through the NYU VPN before SSHing to Greene.  
- The official instructions include example SSH config snippets and commands for greene.hpc.nyu.edu and dtn.hpc.nyu.edu as well as VPN and gateway options: https://sites.google.com/nyu.edu/nyu-hpc/accessing-hpc?authuser=0#h.7t97br4zzvip.

## Clone in $HOME

After logging into Greene, `cd` into your home directory (`cd $HOME`). You must clone your fork into `$HOME` only (not scratch or archive). This ensures subsequent scripts and paths resolve correctly on the cluster. Since this is a private repository, you need to authenticate with GitHub. You have two options:

### Option A: Via VS Code (Recommended)
The easiest way to avoid managing keys manually is to configure **VS Code Remote SSH**. If set up correctly, VS Code forwards your local credentials to the cluster.
- Follow the [NYU HPC VS Code guide](https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/vs-code) to set up the connection.

> **Tip:** Once connected to Greene in VS Code, you can clone directly without using the terminal:
> 1. **Sign in to GitHub:** Click the "Accounts" icon (user profile picture) in the bottom-left sidebar. If you aren't signed in, click **"Sign in with GitHub"** and follow the browser prompts to authorize VS Code.
> 2. **Clone the Repo:** Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`), type **Git: Clone**, and select it.
> 3. **Select Destination:** When prompted, select your home directory (`/home/<netid>/`) as the clone location.
>
> For more details, see the [VS Code Version Control Documentation](https://code.visualstudio.com/docs/sourcecontrol/intro-to-git#_clone-a-repository-locally).

### Option B: Manual SSH Key Setup
If you prefer using a standard terminal, you must generate a unique SSH key on the Greene cluster and add it to your GitHub account:
1. **Generate a key:** Run the `ssh-keygen` command on Greene (follow the official [GitHub documentation on generating a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key)).
2. **Add the key to GitHub:** Copy the output of your public key (e.g., `cat ~/.ssh/id_ed25519.pub`) and add it to your account settings (follow the [GitHub documentation on adding a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)).

### Execute the Clone
Once authenticated, run the following commands. Replace `<your-git-ssh-url>` with the SSH URL of your fork (e.g., `git@github.com:YOUR_USERNAME/rob6323_go2_project.git`).
```
cd $HOME
git clone <your-git-ssh-url> rob6323_go2_project
```
*Note: You must ensure the target directory is named exactly `rob6323_go2_project`. This ensures subsequent scripts and paths resolve correctly on the cluster.*
## Install environment

- Enter the project directory and run the installer to set up required dependencies and cluster-side tooling.  
```
cd $HOME/rob6323_go2_project
./install.sh
```
Do not skip this step, as it configures the environment expected by the training and evaluation scripts. It will launch a job in burst to set up things and clone the IsaacLab repo inside your greene storage. You must wait until the job in burst is complete before launching your first training. To check the progress of the job, you can run `ssh burst "squeue -u $USER"`, and the job should disappear from there once it's completed. It takes around **30 minutes** to complete. 
You should see something similar to the screenshot below (captured from Greene):

![Example burst squeue output](docs/img/burst_squeue_example.png)

In this output, the **ST** (state) column indicates the job status:
- `PD` = pending in the queue (waiting for resources).
- `CF` = instance is being configured.
- `R`  = job is running.

On burst, it is common for an instance to fail to configure; in that case, the provided scripts automatically relaunch the job when this happens, so you usually only need to wait until the job finishes successfully and no longer appears in `squeue`.

## What to edit

- In this project you'll only have to modify the two files below, which define the Isaac Lab task and its configuration (including PPO hyperparameters).  
  - source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py  
  - source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env_cfg.py
PPO hyperparameters are defined in source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/agents/rsl_rl_ppo_cfg.py, but you shouldn't need to modify them.

## How to edit

- Option A (recommended): Use VS Code Remote SSH from your laptop to edit files on Greene; follow the NYU HPC VS Code guide and connect to a compute node as instructed (VPN required off‑campus) (https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/vs-code). If you set it correctly, it makes the login process easier, among other things, e.g., cloning a private repo.
- Option B: Edit directly on Greene using a terminal editor such as nano.  
```
nano source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py
```
- Option C: Develop locally on your machine, push to your fork, then pull changes on Greene within your $HOME/rob6323_go2_project clone.

> **Tip:** Don't forget to regularly push your work to github

## Launch training

- From $HOME/rob6323_go2_project on Greene, submit a training job via the provided script.  
```
cd "$HOME/rob6323_go2_project"
./train.sh
```
- Check job status with SLURM using squeue on the burst head node as shown below.  
```
ssh burst "squeue -u $USER"
```
Be aware that jobs can be canceled and requeued by the scheduler or underlying provider policies when higher-priority work preempts your resources, which is normal behavior on shared clusters using preemptible partitions.

## Where to find results

- When a job completes, logs are written under logs in your project clone on Greene in the structure logs/[job_id]/rsl_rl/go2_flat_direct/[date_time]/.  
- Inside each run directory you will find a TensorBoard events file (events.out.tfevents...), neural network checkpoints (model_[epoch].pt), YAML files with the exact PPO and environment parameters, and a rollout video under videos/play/ that showcases the trained policy.  

## Download logs to your computer

Use `rsync` to copy results from the cluster to your local machine. It is faster and can resume interrupted transfers. Run this on your machine (NOT on Greene):

```
rsync -avzP -e 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' <netid>@dtn.hpc.nyu.edu:/home/<netid>/rob6323_go2_project/logs ./
```

*Explanation of flags:*
- `-a`: Archive mode (preserves permissions, times, and recursive).
- `-v`: Verbose output.
- `-z`: Compresses data during transfer (faster over network).
- `-P`: Shows progress bar and allows resuming partial transfers.

## Visualize with TensorBoard

You can inspect training metrics (reward curves, loss values, episode lengths) using TensorBoard. This requires installing it on your local machine.

1.  **Install TensorBoard:**
    On your local computer (do NOT run this on Greene), install the package:
    ```
    pip install tensorboard
    ```

2.  **Launch the Server:**
    Navigate to the folder where you downloaded your logs and start the server:
    ```
    # Assuming you are in the directory containing the 'logs' folder
    tensorboard --logdir ./logs
    ```

3.  **View Metrics:**
    Open your browser to the URL shown (usually `http://localhost:6006/`).

## Debugging on Burst

Burst storage is accessible only from a job running on burst, not from the burst login node. The provided scripts do not automatically synchronize error logs back to your home directory on Greene. However, you will need access to these logs to debug failed jobs. These error logs differ from the logs in the previous section.

The suggested way to inspect these logs is via the Open OnDemand web interface:

1.  Navigate to [https://ood-burst-001.hpc.nyu.edu](https://ood-burst-001.hpc.nyu.edu).
2.  Select **Files** > **Home Directory** from the top menu.
3.  You will see a list of files, including your `.err` log files.
4.  Click on any `.err` file to view its content directly in the browser.

> **Important:** Do not modify anything inside the `rob6323_go2_project` folder on burst storage. This directory is managed by the job scripts, and manual changes may cause synchronization issues or job failures.

## Project scope reminder

- The assignment expects you to go beyond velocity tracking by adding principled reward terms (posture stabilization, foot clearance, slip minimization, smooth actions, contact and collision penalties), robustness via domain randomization, and clear benchmarking metrics for evaluation as described in the course guidelines.  
- Keep your repository organized, document your changes in the README, and ensure your scripts are reproducible, as these factors are part of grading alongside policy quality and the short demo video deliverable.

## Resources

- [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/ecosystem.html) — Everything you need to know about IsaacLab, and more!
- [Isaac Lab ANYmal C environment](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c) — This targets ANYmal C (not Unitree Go2), so use it as a reference and adapt robot config, assets, and reward to Go2.
- [DMO (IsaacGym) Go2 walking project page](https://machines-in-motion.github.io/DMO/) • [Go2 walking environment used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/tasks/go2_terrain.py) • [Config file used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/cfg/task/Go2Terrain.yaml) — Look at the function `compute_reward_CaT` (beware that some reward terms have a weight of 0 and thus are deactivated, check weights in the config file); this implementation includes strong reward shaping, domain randomization, and training disturbances for robust sim‑to‑real, but it is written for legacy IsaacGym and the challenge is to re-implement it in Isaac Lab.
- **API References**:
    - [ArticulationData (`robot.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationData) — Contains `root_pos_w`, `joint_pos`, `projected_gravity_b`, etc.
    - [ContactSensorData (`_contact_sensor.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.ContactSensorData) — Contains `net_forces_w` (contact forces).

---
Students should only edit README.md below this line.


## Project Execution Guide

This section details how to execute the training pipeline on the NYU HPC server and retrieve the results.

### 1. Running the Training Job
Once you have cloned the repository and navigated to the `rob6323_go2_project` directory on the NYU HPC server, you can initiate the training process. 

We have provided a helper script to streamline the submission. Run the following command to submit the job to the HPC scheduler:

./train.sh


### 2. Retrieving Results

After the training job has completed successfully, the logs and results will be stored in the logs directory on the server. To analyze these results locally, you can transfer them to your personal machine.

Open a terminal on your local device (not the HPC terminal) and run the following command. Replace <user> with your NYU NetID:

```
scp -r <user>@dtn.hpc.nyu.edu:/home/<user>/rob6323_go2_project/logs ./
```

## 3. Configuration & Reward Tuning

To improve the robot's locomotion quality and stability, we modified `rob6323_go2_env_cfg.py`.
Included in the repo right next to this file is the baseline file as well named: `rob6323_go2_env_cfg_baseline_policy.py'
Below is a detailed breakdown of the changes made from the baseline configuration to our optimized setup.

### 3.1 Sharpening Command Tracking
We increased the weights for velocity and yaw commands to force the policy to prioritize following user inputs more strictly. The baseline agent was often too "lazy," drifting from the target velocity.

* **`lin_vel_reward_scale`**: Increased from **1.0 to 2.0**.
    * *Reasoning:* Doubles the incentive for tracking $v_x, v_y$. This reduces steady-state error when the robot is commanded to move at specific speeds.
* **`yaw_rate_reward_scale`**: Increased from **0.5 to 1.0**.
    * *Reasoning:* Doubles the reward for matching the desired turning rate, resulting in sharper, more responsive turns.

### 3.2 Enhanced Stability & Smoothness
The initial configuration allowed for too much high-frequency vibration in the joints and unnecessary body roll/pitch. We increased penalties to dampen these behaviors.

* **`dof_vel_reward_scale`**: Increased penalty significantly from **-0.0001 to -0.005**.
    * *Reasoning:* A 50x stronger penalty on joint velocities. This suppresses high-frequency jitter and encourages smoother leg movements.
* **`ang_vel_xy_reward_scale`**: Increased penalty from **-0.001 to -0.005**.
    * *Reasoning:* A 5x stronger penalty on non-yaw angular velocities. This forces the robot to keep its trunk flat (stable roll and pitch) while moving.
* **`torque_l2_reward_scale`**: Added/Modified to **-1e-4**.
    * *Reasoning:* Penalizes high torque usage ($||\tau||^2$), encouraging energy-efficient motions and preventing the policy from banging against torque limits.

### 3.3 Physical Feasibility & Safety (New Constraints)
We introduced specific penalties to ensure the gait is physically realistic and safe for hardware deployment (Sim-to-Real considerations).

* **`feet_slip_reward_scale`**: New penalty set to **-0.1**.
    * *Reasoning:* Penalizes foot velocity when the foot is in contact with the ground. This minimizes "ice-skating" behavior, ensuring solid traction.
* **`knee_collision_reward_scale`**: New penalty set to **-0.02**.
    * *Reasoning:* Explicitly penalizes collisions between the robot's own links (knees/hips), preventing self-intersection which causes physics glitches in sim and damage in real hardware.

### 3.4 Retained Gait Structuring
The following Raibert-based rewards were kept consistent with the baseline to maintain the fundamental trotting gait structure:
* `raibert_heuristic_reward_scale` (-10.0)
* `feet_clearance_reward_scale` (-30.0)
* `tracking_contacts_shaped_force_reward_scale` (4.0)

### 4. Environment Class Updates: `__init__`

The initialization logic in `Rob6323Go2Env` was updated to support the new reward functions and sensors required for our improved training config.

#### 4.1 Expanded Episode Logging
We updated the `_episode_sums` dictionary to include storage for the new reward terms we introduced.
* **Added `torque_l2`**: Tracks the sum of squared torques for the energy efficiency penalty.
* **Added `feet_slip`**: Tracks the penalty for feet sliding while in contact with the ground.
* **Added `knee_collision`**: Tracks the penalty for non-foot collisions (knees/calves).

#### 4.2 Enhanced Collision Detection Setup
In the previous implementation, the code only looked for "thigh" bodies to detect unwanted collisions. We expanded this to include "calf" bodies as well, ensuring that the robot learns to avoid hitting its lower legs against the ground or obstacles.

* **Old Code**:
    ```python
    self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*thigh")
    ```
* **New Code**:
    ```python
    # Search for both thigh and calf bodies for comprehensive collision penalization
    self._knee_body_ids: list[int] = []
    for pattern in ["thigh", "calf"]:
        ids, _ = self._contact_sensor.find_bodies(f".*{pattern}.*")
        for b in ids:
            self._knee_body_ids.append(b)
    ```

#### 4.3 State Buffer for Torques
We added a new buffer, `self.last_torques`, initialized to zero. This allows us to access the torques applied in the previous step, which is necessary for calculating the $L^2$ torque penalty (`torque_l2_reward_scale`) during the step function.

### 5. Control & Action Updates: `_apply_action`

We modified the `_apply_action` method to not only apply the computed torques to the robot but also store them. This is a prerequisite for the energy efficiency penalty introduced in the reward function.

* **Torque Storage**:
    * *Change:* We assigned the computed PD control output to `self.last_torques` before sending it to the robot.
    * *Reasoning:* This allows the `_get_rewards` function to access the exact torque values applied during the step to calculate the $L^2$ penalty ($||\tau||^2$), encouraging the policy to solve the task using minimal energy.

### 6. Reward Calculation Overhaul: `_get_rewards`

The `_get_rewards` function was significantly refactored to incorporate the new physics constraints and fix order-of-operation issues regarding history buffers.

#### 6.1 Action Smoothness & History Logic
In the previous implementation, the history buffer `self.last_actions` was updated (rolled) *after* the reward calculation but potentially *before* some diagnostic metrics were computed (depending on helper function implementation).

* **Explicit Difference Calculation**: We now compute the first and second-order finite differences (velocity and acceleration of actions) **before** modifying the history buffer.
* **Reasoning**: This guarantees that both the reward term `rew_action_rate` and the diagnostic metric `diag_jerk` are derived from the exact same state transitions, ensuring that our logging accurately reflects the agent's behavior.

#### 6.2 New Physical Constraint Terms
We integrated the new penalty terms directly into the reward composition:

1.  **`torque_l2`**:
    * Calculated as `torch.sum(self.last_torques ** 2, dim=1)`.
    * Prevents the policy from "bang-bang" control strategies where it maxes out actuators to achieve stability.
2.  **`feet_slip`**:
    * Calls the new `_reward_feet_slip()` helper.
    * Penalizes non-zero velocity for any foot currently in contact with the ground.
3.  **`knee_collision`**:
    * We replaced the opaque `_metric_knee_collision` helper with an explicit check inside the main loop.
    * It checks the net contact forces on the defined "knee" bodies (thighs + calves). If the force exceeds **5.0 N**, it registers as a collision.

#### 6.3 Consolidated Diagnostics
We streamlined the diagnostic logging process. Instead of calling separate helper functions (like `_metric_slip_and_toedrag` or `_metric_posture`) which re-calculate values, we now reuse the intermediate variables computed for the rewards.

* **Efficiency**: This reduces computational overhead by avoiding redundant math.
* **Consistency**: `diag_slip` is now exactly equal to the unscaled `rew_feet_slip`, ensuring that what we plot in the logs is exactly what the agent is being penalized for.

### 7. Termination Conditions: `_get_dones`

We retained the robust termination logic from the baseline implementation to ensure the robot learns safe behaviors. While the code was reformatted for better readability, the core safety checks remain active:

* **Base Contact**: The episode terminates if the robot's body (chassis) hits the ground with significant force (> 1.0 N). This prevents the robot from learning to "belly crawl."
* **Orientation Check**: The episode terminates if the robot flips over (`projected_gravity_b[:, 2] > 0`), defining a fall.
* **Height Constraint**: The episode ends if the base height drops below a minimum threshold (`cfg.base_height_min`), ensuring the robot maintains a standing posture.

### 8. Reset Logic: `_reset_idx`

The reset function was updated to ensure that all state buffers—including the newly added ones—are correctly cleared when an episode restarts.

#### 8.1 Clearing Torque History
A critical addition to this function was the resetting of the torque buffer.

* **Change**: Added `self.last_torques[env_ids] = 0.0`.
* **Reasoning**: Since we introduced `torque_l2` penalties in the reward function, it is essential to zero out the torque history upon reset. This prevents "phantom" torque values from a previous crash or timeout from contaminating the reward calculation of the first step in the new episode.

#### 8.2 Logging Structure
We streamlined the logging loop to clearly distinguish between optimization targets (rewards) and analytical metrics (diagnostics).

* **Logic**:
    ```python
    if key.startswith("diag_"):
        prefix = "Episode_Diag/"
    else:
        prefix = "Episode_Reward/"
    ```
* **Impact**: This ensures that when viewing TensorBoard or log files, the "Episode_Diag" folder contains only the metrics we use for debugging (like jerk or slip), while "Episode_Reward" contains the actual terms driving the policy update.

### 9. Gait Quality Helpers: `_reward_feet_and_contacts`

We enhanced the foot tracking helper to calculate an additional diagnostic metric: **Toe Drag**.

#### 9.1 Diagnostic Return Value
The function signature was updated to return a third tensor, `diag_toe_drag`.
* **Old**: Returned `(clearance_reward, contact_timing_reward)`
* **New**: Returns `(clearance_reward, contact_timing_reward, toe_drag_metric)`

#### 9.2 Measuring Toe Drag
We added logic to specifically measure how often the robot fails to lift its feet high enough during the swing phase.

* **Calculation**:
    ```python
    swing_mask = 1.0 - self.desired_contact_states
    toe_deficit = torch.clamp(desired_height - rel_foot_height, min=0.0)
    diag_toe_drag = torch.sum(swing_mask * toe_deficit, dim=1)
    ```
* **Reasoning**: While `rew_feet_clearance` penalizes *any* deviation from the target trajectory (whether too high or too low), `diag_toe_drag` isolates the specific failure mode where the foot is **too low** during swing. This is a critical metric for real-world deployment, as toe dragging is the primary cause of stumbling on uneven terrain.


### 10. Refactoring: Removal of Redundant Metrics
To optimize the environment's performance, we removed the following standalone diagnostic functions:

* `_metric_slip_and_toedrag`
* `_metric_foot_timing`
* `_metric_jerk`
* `_metric_posture`
* `_metric_knee_collision`

**Reasoning:** In the previous implementation, these functions re-calculated physical quantities (such as action differences or contact forces) that were already being computed inside the `_get_rewards` loop. By moving this logic inline (as detailed in Section 6), we eliminated redundant calculations and ensured that our diagnostic logs (e.g., `diag_jerk`) exactly match the terms used in the reward function.

### 11. New Helper: `_reward_feet_slip`
We replaced the old slip metric logic with a dedicated reward helper function. This function penalizes horizontal foot velocity specifically when the foot is in the stance phase (contacting the ground).

```python
    def _reward_feet_slip(self) -> torch.Tensor:
        """
        Calculates the penalty for feet sliding while in contact.
        Also used for the 'diag_slip' metric.
        """
        contact_mask = self.desired_contact_states
        
        # Get foot velocities in world frame
        foot_vel_w = self.robot.data.body_lin_vel_w[:, self._feet_ids]
        
        # We only care about horizontal velocity (x, y)
        foot_vel_xy_mag = torch.norm(foot_vel_w[:, :, :2], dim=-1)
        
        # Reward: penalize velocity when contact is desired
        rew_slip = torch.sum(contact_mask * foot_vel_xy_mag, dim=1)
        
        return rew_slip



