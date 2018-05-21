#!/bin/bash

# first value is total duration, second value is gap between images being taken
# all time values in milliseconds

raspistill -t 45000 -tl 1250 -o image%04d.jpg
