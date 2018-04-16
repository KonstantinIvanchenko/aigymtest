import numpy as np
#set of auxilary functions

#set of functions to downscale image
def grayscale_img(img):
    #img[:50, :50] = np.ma.masked
    img = np.mean(img, axis=2 ).astype(np.uint8)

    #return np.mean(img, axis=2 ).astype(np.uint8)
    return img

#downscale the size
def downsample_img(img):
    #return np.resize(img, (int(np.size(img,axis = 0)/2), int(np.size(img,axis = 1)/2)))
    return img[::2, ::2]

#change reward to [-1,0,1]
def transform_reward(reward):
    return np.sign(reward)

#
def epsilon_get_linear(numberOfIterations, strength):
    if numberOfIterations >= strength:
        return 0.1
    else:
        return -(1/strength)*numberOfIterations + 1.0

def set_max_action_to_one(action):
    max_index = np.argmax(action, axis=1)
    np.ndarray.fill(action, 0)
    action[0, max_index] = 1
    return action