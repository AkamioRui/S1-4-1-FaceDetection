# implicit way
- basically immediately use what plot you want. *plt.plot*, *plt.imshow*, etc
- if you want to put it adjacent, only then start prefixing each plot with *plt.subplot*
- if you want to put it on other figure, only then start prefixing each subplot with *plt.figure*



# figure
- the actual window size is based on how many pixel
    - *figsize* is how many inches the whole figure is being interpreted, this determine the size of font, not the actual visible size of the figure. usually, if the figure will have w x h pixel, `figsize = w/dpi, h/dpi`
    - *dpi* pixel per inch
- *facecolor* = (r,g,b,a) 0-1. alpha doesnt show in window, but does in image(png/jpg)
- *layout* to manage overlap for labels

# math text
- *$...$* surround with $$
- *_{...}* for subscript *^{...}* for superscript.
    - used after *A*(a variable), *\int*, *\sum*, *\lim*. e.g. `\lim_{...\to...}`
- < \leq = \approx \neq \equiv
- *\exists* *\forall* *\therefore* *\rightarrow* *\leftrightarrow*
- *\cap* *\cup* *\supset* *\subset* *\in* *\notin*

# color
- imshow will treat data as colormap if NxMx1, RGB if NxMx3, RGBA if NxMx4
- custom color map
```py

boundary = [x1,x2,x3,x4,x5,x6] #this create 5 bin
cmap = ListedColormap([c1,c2,c3,c4,c5]) 
norm = BoundaryNorm(boundary,cmap.N) # return a function: data -> cmap_idx. cmap[cmap_idx]
plt.imshow(data, cmap=cmap, norm=)
```
- customizing details shown when hovering 