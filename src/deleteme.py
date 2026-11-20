def render_result(data:torch.Tensor,target:torch.Tensor,extent:tuple[int,int,int,int],subplot:tuple[int,int,int]):
    data = data.detach().to(device='cpu')
    target = target.detach().to(device='cpu')

    # renderingz
    plt.subplot(*subplot)
    im = plt.imshow(
        data,
        extent=extent,

        cmap = (cmap:=ListedColormap([
            *[((5-x)/5,0,0) for x in range(5)],
            (1,1,1),
            *[(0,0,(x+1)/5) for x in range(5)],
        ])),
        norm=BoundaryNorm(
            bounds:=[
                *[(5-x)/5*data.min() for x in range(5)] ,
                data.min()/100, data.max()/100, 
                *[(x+1)/5*data.max() for x in range(5)]
            ]
            ,cmap.N
        ),
        
        
    )
    plt.colorbar(im,ax=plt.gca(),ticks=bounds)

    
    plt.imshow(
        target,
        extent= extent,
        cmap= ListedColormap([(0,0,0,0),(0,1,0,1)]),
        norm= BoundaryNorm([0,0.5,1],2,),
    )

    def format_coord(x,y):
        """ 
        extent = (-2, 2, -2, 2) 

        """
        # x,y to data's[idx_x,idx_y]
        N = data.shape[0]
        idx_x = round((x-extent[0])/(extent[1]-extent[0])*(N-1))
        idx_y = round((extent[3]-y)/(extent[3]-extent[2])*(N-1))
        value = float(data[idx_y,idx_x])
        return f'x={x:.2f}, y={x:.2f}\n [{value:.2g}]'
    plt.gca().format_coord = format_coord
    plt.gca().format_cursor_data = lambda data: ''