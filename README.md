# YieldMap

1. What you can get from the output: 
        1). Accuracy.txt, 'a+', it shows the accuracy, recalls and precisions of each kernel map(one kernel map for each step).
Every three lines corresponds to one line in other files. Just for you to watch the results.
        2). flightpath.txt, 'a+', it shows the flight path(visible zones). One line for each kernel map. The format is as follows:
"754,796,838,839,881,". For 754, it's (row) times (the width of the whole map) plus (col).
        3). pred_map.txt, 'a+', it shows the predicted yield map for each kernel map. One line for each kernel map. The format is
as follows: "0.2195161134004593,0.23554691672325134,0.47628548741340637,0.48634713888168335,...". The raw predictions from the NN
models. The length of each line depends on the size of the kernel map. Length equals to (rows) times (cols).
        4). true_map.txt, 'a+', it shows the ground truth data of the yield map of each kernel map. One line for each kernel map.
The format is as follows: "0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,...". The results from the GE. 0 for the good class and 1 for the bad.
It corresponds to the pred_map.txt. The length of each line is the same as the corresponding line in the pred_map.txt.
        5). neighborzone_truth.txt, 'a+', it shows the ground truth data for all the next available steps. Each line corresponds 
to the line in pred_map.txt and true_map.txt. The format is as follows: "469: 2, 468: 0, 427: 1". The first "469: 2" is info. 
"469" is the current zone that the drone stays, "2" means it has 2 available neighbors that can be fly to. I don't add other neighbor
zones that aren't reachable. "468" and "427" are the available zones for choosing the next step. "0" and "1" are the class they belong
to. Same as above, 0 for good and 1 for bad.
        6). yield_map.txt, the whole yield map for the whole field after one flight path completes. Since you don't need it. It 
just contains the predictions of all the visible zones. You can just ignore it.

2. What you can change:
        I wrote commits in the python file. It's better to commit there. Basically what you need to change is the size of the kernel 
map and the least steps for each path.

3. How to run it for 10,000 times. I have a stupid way to do this. There's a duplicate.sh file. You can change the parameters in it.
Since each line in all files that you need is one kernel map. You can save all the result in these same files.
