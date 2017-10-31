#!/usr/bin/env python
#coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    if len(sys.argv) == 3:
        plot_error(sys.argv[1], int(sys.argv[2]))
    elif len(sys.argv) == 2:
        plot_error(sys.argv[1])
    else:
        print "specify *.log file."

def plot_error(file_name, max_loop = -1):
    
    with open(file_name, 'r') as fin:
        lines = fin.readlines()

    loops = []
    errors = []
        
    for l in lines:
        if l.find("RMS error after ") == 0:
            tmp = l.find("th training")
            if tmp != -1:
                e = int(l[16:tmp])
                if max_loop < 0:
                    loops.append(e)
                    errors.append(float(l[l.find(':')+1 : ]))
                elif e <= max_loop:
                    loops.append(e)
                    errors.append(float(l[l.find(':')+1 : ]))

                    
    # plt.plot(loops, errors, marker = 'o', markersize = 4)
    plt.plot(np.array(loops) + 1, errors, marker = '.')
    plt.xscale("log")
    plt.xlabel("loop")
    plt.ylabel("rms error [Nm]")
    max_x = max(loops)
    
    xticks = []
    i = 0
    while True:
        xticks.append(10**i)
        if max_x <= 10**i:
            break
        i += 1
        
    plt.xticks(xticks)
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
