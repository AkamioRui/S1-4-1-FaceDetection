# analysis method
- find the kernel (input space that doesnt change the output) of a cell at a convolution 
    - abuse negative input to RELU?
    - (space too big) try iterating over all input that output high activation
- displaying the kernel     
    - certainly cant just show all since it will be a N dimensional object
    - instead describe it using the nature of each dimension. e.g. the kernel has the "scale" invariant, "red to green" invariant
    - ideally, the nth quality choosen to represent a dimension should be orthogonal, that is to say, one quality's value doesnt effect the range of possible other quality's value. 
        - thus need to find a way to map a nth dimensional unit cube into the space (not any arbitrary space, but a kernel space for a stack of convolution)
- calculate the dimension 
    - in conclusion, the dimension will be either the RF or RF-1 (so if it depend on 4 rgb pixel, it has 12 or 11 dimension)
    - pick a random point in the kernel, generate as much othrogonal neighbor, using derivative
        - you will discover, the derivative of that cell will just be a vector dotted with the input pixel (albeit, the vector will change depending on the location, to account for the unlinearity) so =0 ment it lost at most 1 dimension [1]
- the `xyTest` experiment show that the convultion can only approximate 
    - it approximate by deviding the input space into region using plane, then each plane will be evaluated by a linear function. 
    - it's approximation is pointwise continuous
    - given that approximating a circle using a line will take many linear function, but just 1 circle function. what if there is a model that can interpret a weight classify it into a list of handmade function
        - but this list of function will take more than just 2 input. should be a simple function that is hard to have linear approximation
    - while that l[n+1] will have subdivision decided by l[n], can you construct a function that
        - have any arbitratry direction per region
        - as long as it is continous
- smoother model
    - try gaussian `blur the model`, making the derivative view a smoother version of the real model that still keep the location of the minimum the same. do this `only whenever adam changes` direction to 
    - 
- find the dot product of the gradient with the previous one, to give an overview of how smooth the function is
# small model experiment
- have a model take (x,y) and try to guess if it will be true or not, the condition for true can be inside a circle r=3, 

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

# literature review
Hanin & Rolnick, “Complexity of Linear Regions in Deep Networks” (2019), and “Deep ReLU Networks Have Surprisingly Few Activation Patterns” (2019). These characterize every activation pattern (linear region) per layer across the whole input space.
Montúfar et al., “On the Number of Linear Regions of Deep Neural Networks” (2014).