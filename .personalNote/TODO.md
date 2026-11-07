# analysis method
1. try iterating over all input that output high activation

# bottom up analysis
train and analyze per layer
input --> 1 3x3conv --> 1 1x1conv --> output(w/o anchor)
input --> 1 3x3conv --> 1 1x1conv --> output(w/ anchor)

# top down analysis
from a premade model

# hypothesis
the first block try to detect line and circle
next block, try to detect a combination of the previous block feature
    3x3conv means it also sees other cell's feature, imagine a line with a circle at the end, then the 3x3conv will have weight at  [current cell's, line channel] and [ cell above's, circle channel]

    following this, the first block output will have line for each angle at different channel, that means you can pass it to a block with less channel so that it encodes it into degrees and possibly length
