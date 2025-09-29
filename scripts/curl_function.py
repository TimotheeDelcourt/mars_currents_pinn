import torch



def curl_differentiable(Input, Output):
    """
    Compute curl while maintaining computational graph for higher-order derivatives
    """
    # Use torch.autograd.grad instead of .backward() to maintain graph
    
    # Compute all partial derivatives using the same logic as the working version
    # ∂Bx/∂(x,y,z)
    grad_Bx = torch.autograd.grad(Output[:,0].sum(), Input, create_graph=True, retain_graph=True)[0]
    dBx_dx, dBx_dy, dBx_dz = grad_Bx[:,0], grad_Bx[:,1], grad_Bx[:,2]
    
    # ∂By/∂(x,y,z)  
    grad_By = torch.autograd.grad(Output[:,1].sum(), Input, create_graph=True, retain_graph=True)[0]
    dBy_dx, dBy_dy, dBy_dz = grad_By[:,0], grad_By[:,1], grad_By[:,2]
    
    # ∂Bz/∂(x,y,z)
    grad_Bz = torch.autograd.grad(Output[:,2].sum(), Input, create_graph=True, retain_graph=True)[0] 
    dBz_dx, dBz_dy, dBz_dz = grad_Bz[:,0], grad_Bz[:,1], grad_Bz[:,2]
    
    # Curl formula: (∂Bz/∂y - ∂By/∂z, ∂Bx/∂z - ∂Bz/∂x, ∂By/∂x - ∂Bx/∂y)
    curl_x = dBz_dy - dBy_dz
    curl_y = dBx_dz - dBz_dx  
    curl_z = dBy_dx - dBx_dy
    
    return torch.stack((curl_x, curl_y, curl_z), dim=1)