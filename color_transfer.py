import numpy as np 

def compute_mean_cov(img, eps=1e-5):
    img_mean = np.mean(img, axis=(0, 1))
    img = img - img_mean
    img = img.transpose(2, 0, 1).reshape(3, -1)
    img_covariance = img.dot(img.T) / img.shape[1] + eps * np.eye(img.shape[0])
    return (img_mean, img_covariance, img)

def pixel_transformation(mode, style, content):
    style_mean, style_covariance, style_transpose = compute_mean_cov(style)
    content_mean, content_covariance = compute_mean_cov(content)[0], compute_mean_cov(content)[1]
    # Implementing Chloesky Decomposition 
    if mode == 'cholesky':
        Lc = np.linalg.cholesky(content_covariance)
        Ls = np.linalg.cholesky(style_covariance)
        A = np.dot(Lc, np.linalg.inv(Ls))
    # Implementing Image Analogies Decomposition 
    elif mode == 'image_analogies':
        ws, vs = np.linalg.eigh(style_covariance)
        wc, vc = np.linalg.eigh(content_covariance)
        style_cov_sqr = np.dot(np.dot(vs, np.sqrt(np.diag(ws))), vs.T)
        content_cov_sqr = np.dot(np.dot(vc, np.sqrt(np.diag(wc))), vc.T)
        A = np.dot(content_cov_sqr, np.linalg.inv(style_cov_sqr))
    # producing a new style image as input to the neural style transfer algorithm
    resized_transformed_style = np.dot(A, style_transpose).reshape(style.transpose(2, 0, 1).shape).transpose(1, 2, 0)
    new_style = resized_transformed_style + content_mean
    return new_style


    

