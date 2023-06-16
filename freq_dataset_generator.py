import torch
from keras.datasets import mnist, fashion_mnist
import os
import numpy as np

def filter_percent_frequency(batched_input,mode,filter_rate = 0.60):
    batched_input = batched_input.to(device="cuda")
    dft_b_input = torch.fft.fft2(batched_input)
    dft_b_input = torch.fft.fftshift(dft_b_input)
    h, w = dft_b_input.size()[-2:] # height and width
    cy, cx = int(h/2), int(w/2) # centerness
    if(mode=="LFC"):
        rh, rw = int((1-filter_rate) * cy), int((1-filter_rate) * cx) # filter_size
        # the value of center pixel is zero.
        dft_b_input[:,rh:-rh, rw:-rw] = 0
    elif(mode == "HFC"):
        rh, rw = int(filter_rate * cy), int(filter_rate * cx) # filter_size
        dft_b_input[:,:rh,:] = 0
        dft_b_input[:,-rh:,:] = 0
        dft_b_input[:,:,:rw] = 0
        dft_b_input[:,:,-rw:] = 0
    else:
        assert False, "mode not valid"
    dft_b_input = torch.fft.ifftshift(dft_b_input)
    dft_b_input = torch.fft.ifft2(dft_b_input)
    return torch.real(dft_b_input)

def modify_get_dataset(loader,mode,filter_percent):
    mod_x = []
    ys = []
    for X_batch, y_batch in loader:
        X_batch = filter_percent_frequency(X_batch,mode,filter_percent)
        mod_x.extend(X_batch.cpu().numpy())
        ys.extend(y_batch.cpu().numpy())
    
    return np.array(mod_x),np.array(ys)


if __name__ == '__main__':
    # fashion_mnist , mnist
    datasetname = "mnist"
    mode = "HFC"
    filter_percent = 0.30

    for filter_percent in [0.3,0.5,0.65,0.8]:
        if(datasetname == "mnist"):
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
        elif(datasetname == "fashion_mnist"):
            (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        
        loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), shuffle=False, batch_size=256)
        (mod_X_train, mod_y_train) = modify_get_dataset(loader,mode,filter_percent)

        assert mod_X_train.shape == X_train.shape and (mod_y_train == y_train).all(), "Error in creating train set"
        
        # loader = torch.utils.data.DataLoader(list(zip(X_test, y_test)), shuffle=False, batch_size=256)
        # (mod_X_test, mod_y_test) = modify_get_dataset(loader,mode,filter_percent)

        # assert mod_X_test.shape == X_test.shape and (mod_y_test == y_test).all(), "Error in creating test set"

        save_folder = "data/custom_datasets/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        with open(save_folder+str(datasetname)+"_"+str(mode)+"_FP_"+str(filter_percent)+".npy", 'wb') as file:
                np.savez(file, X_train=mod_X_train,y_train=mod_y_train,X_test=X_test,y_test=y_test)
    
    print("Completed creation of dataset")
