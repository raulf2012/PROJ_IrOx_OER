#!/bin/bash

# #########################################################
# Will create a file job.out.short from job.out that includes the first 10,000 lines and the last 300 lines from job.out.
# Useful in cases where millions of repetitive lines are appended to job.out from VASP
# #########################################################

echo "PWD"
echo $PWD

# Create new job.out.short file
head -n 10000 job.out > job.out.short
echo "\n \n \n \n \n \n \n " >> job.out.short
echo "The middle contents of job.out were removed" >> job.out.short
echo "The middle contents of job.out were removed" >> job.out.short
echo "\n \n \n \n \n \n \n " >> job.out.short
tail -n 300 job.out >> job.out.short

# Remove old job.out
rm job.out
