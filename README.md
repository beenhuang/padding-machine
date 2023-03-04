# Break-Pad

This repository contains the source code for the WF defense described in the following paper:
**`Break-Pad: Effective Padding Machines for Tor with Break Burst Padding`**


### Setup  
1. **Modified Tor**: download modified Tor software at ```https://pan.baidu.com/s/1-b3BHwM7Me1z8RXFgls2_A``` (access code: h7ew), and run ```make and make install``` command to install Tor.
2. **Python Script**: Clone this repo using  ```git clone https://github.com/beenhuang/padding-machine.git``` command.
3. **GoodEnough Dataset**: download Pulls's GoodEnough dataset at ```https://dart.cse.kau.se/goodenough/goodenough-feb-2020.zip``` and put the datset in the **"data"** folder.

### Run Simulation and Evaluate the defended traces
1. **Run Simulator:** ```./simulation/run_simulation.py --in standard --out "defended trace" --machine "machine_name"```
2. **Bandwidth Overhead:** ```./evaluation/overhead.py --in "defended trace"  --out "result"```
3. **CUMUL:** ```./evaluation/cumul/run_cumul.py --in "defended trace"  --out "result"```
4. **k-FP:** ```./evaluation/k-FP/run_kFP.py --in "defended trace"  --out "result"```
5. **DF:** ```./evaluation/df/run_df.py --in "defended trace"  --out "result"```

OR, You can run the ```october.sh``` file, and other files are in the **"run"** directory. 

### Parameter Tuning
1. **Get Original Trace:** run the ```./simulation/origin-trace.py --in standard --out "original trace"  --maxlength 5000``` command at the base directory. 
2. **Distirbution Fitting**: run  the ```./burst.py --in "original trace" --out "result"``` command in the **"hyperparameter"** folder.


### Contact
email: beenhuang@126.com

Any discussions, suggestions, and questions are welcome!
