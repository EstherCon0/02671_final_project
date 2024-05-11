import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import scipy.io
import tensorflow as tf

###going to test it on 5 different resultions


class SpectralConv2d(nn.Module):
    def __init__(self,
                 in_channels,   # Number of input channels
                 out_channels,  # Number of output channels
                 modes1,        # Number of Fourier modes to multiply in the first dimension
                 modes2):       # Number of Fourier modes to multiply in the second dimension
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)


def prep_data_from_u(x_train__):

    N1,N2 = x_train__.shape

    x_org = np.reshape(x_train__, (N1, N2, 1))
    x_org = np.transpose(x_org, (2, 0, 1))
    x_org = np.float32(x_org)


    y_train = np.zeros_like(x_org)
    y_train = np.float32(y_train)
    y_train= x_org[:, 1:, :]

    x_train = np.zeros_like(x_org)
    x_train = x_org[:, :-1, :]

    m1,m2,m3 = x_train.shape  

    x_train = np.reshape(x_train, (m2, 1, m3))
    y_train = np.reshape(y_train, (m2, 1, m3))

    x_train = np.expand_dims(x_train, axis=1)
    y_train = np.expand_dims(y_train, axis=1)

    x_train = np.reshape(x_train, (m2, 1, m3,1))
    y_train = np.reshape(y_train, (m2, 1, m3,1))

    return x_train, y_train,x_org

    # Solve the ODE using fft
def rhsBurgers(u, t, kappa, epsilon):
    uhat = fft(u)
    duhat = (1j)*kappa*uhat
    dduhat = -np.power(kappa,2)*uhat
    du = ifft(duhat)
    ddu = ifft(dduhat)
    dudt = -u*du + epsilon*ddu
    return dudt.real


def calc_the_rmse_for_model1_for_a_specific_resolution(N2):
    N = 256+1
    x = np.linspace(-1,1, N)

    tend = 0.5
    t = np.linspace(0, tend, N2)


    dx = x[1] - x[0]

    # Initial condition
    u0 = -np.sin(np.pi*x)


    # Define discrete wavenumbers
    kappa = 2*np.pi*np.fft.fftfreq(N, d=dx)
    epsilon = 0.01/np.pi


    u = odeint(rhsBurgers, u0, t, args=(kappa,epsilon))

    train_portion = 0.8
    N_train = int(train_portion*N2)
    x_train__ = u[:N_train,:]
    x_test__ = u[N_train:,:]

    #Split x and t as well
    t_train = t[:N_train]
    t_test = t[N_train:]

    

    x_train,y_train,x_org = prep_data_from_u(x_train__)


    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)


    # Define the neural network
    #train the model
    model = SpectralConv2d(1, 1, 200, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    n_epochs = 3000


    for epoch in range(n_epochs):
        # Run the forward pass
        y_pred = model(x_train)

        # Calculate the loss
        loss = criterion(y_pred, y_train)

        # Zero the gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')


    y_pred = model(x_train)


    #make iteratix prediction and then calc rmse
    #only predict 30 into the future
    x_test,y_test,x_test_org = prep_data_from_u(x_test__)

    #make them into torch tensor
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test)

    x_train_val = x_test[-1:]
    print(x_train_val.shape)


    y_pred_val = model(x_train_val)
    predictions = []


    for i in range(30):
        x_train_val = y_pred_val
        y_pred_val = model(x_train_val)
        # Save the prediction
        prediction_array = y_pred_val.squeeze().detach().numpy()  # Convert squeezed tensor to NumPy array
        predictions.append(prediction_array)

    predictions = np.array(predictions)

    # Calculate the RMSE for the model
    if len(x_test__[:30]) >= 30:
        rmse = np.sqrt(np.mean((predictions - x_test__[:30])**2))
        # Calculate range of true values
        #value_range = np.max(x_test__[:30]) - (-np.min(x_test__[:30]))  # Range of data: max - min

        # Calculate relative error
        relative_error = np.linalg.norm(predictions - x_test__[:30]) / np.linalg.norm(x_test__[:30])

    else:
        print("The length of the test data is less than 30")
        rmse = 0

    print(f'The RMSE of the model is: {rmse}')
    return rmse,relative_error





def calc_the_rmse_for_model2_for_a_specific_resolution(N2):
    N = 256+1
    x = np.linspace(-1,1, N)

    tend = 0.5

    t = np.linspace(0, tend, N2)


    dx = x[1] - x[0]

    # Initial condition
    u0 = -np.sin(np.pi*x)


    # Define discrete wavenumbers
    kappa = 2*np.pi*np.fft.fftfreq(N, d=dx)
    epsilon = 0.01/np.pi


    u = odeint(rhsBurgers, u0, t, args=(kappa,epsilon))

    train_portion = 0.8
    N_train = int(train_portion*N2)
    x_train__ = u[:N_train,:]
    x_test__ = u[N_train:,:]


    #Split x and t as well
    t_train = t[:N_train]
    t_test = t[N_train:]



    import torch
    ####this is the one for the 10 time instances!!


    # Assuming your data is stored in a tensor called 'data'
    data = torch.tensor(x_train__ ) # Example random data, replace it with your actual data

    # Reshape the data into sequential samples of 10 consecutive time instances
    num_time_instances = data.shape[0]
    num_input_channels = 10

    # Calculate the number of sequential samples
    num_samples = num_time_instances - num_input_channels + 1
    y_train_reshaped = data[num_input_channels - 1:]

    # Reshape the data into (num_samples, num_input_channels, num_features)
    reshaped_data = torch.zeros(num_samples, num_input_channels, data.shape[1])

    for i in range(num_samples):
        reshaped_data[i] = data[i:i+num_input_channels]


    # Check the shape of the reshaped data



    #put an extra dimension on the reshaped data
    reshaped_data = reshaped_data.unsqueeze(1)
    y_train_reshaped = y_train_reshaped.unsqueeze(1)
    y_train_reshaped = y_train_reshaped.unsqueeze(1)


    #reshape the 1 -dimension to be the last
    reshaped_data = reshaped_data.permute(0,2,3, 1)
    y_train_reshaped = y_train_reshaped.permute(0,2,3, 1)
    #y_train_reshaped = y_train_reshaped.permute(0, 2, 1)



    ####run the model

    x_train = reshaped_data
    y_train = y_train_reshaped

    x_train = x_train.float()
    y_train = y_train.float()


    model = SpectralConv2d(10, 1, 240, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()


    n_epochs = 3000
    for epoch in range(n_epochs):
        # Run the forward pass
        y_pred = model(x_train)

        # Calculate the loss
        loss = criterion(y_pred, y_train)

        # Zero the gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')


    

    y_pred = model(x_train)

    x_train_val = x_train[-1:]
    print(x_train_val.shape)


    y_pred_val = model(x_train_val)
    predictions = []


    for i in range(30):

        # Predict using the model
        y_pred_val = model(x_train_val)

        # Replace the last element of the second dimension with y_pred_val
        #x_train_val_2 = torch.zeros_like(x_train_val)
        x_train_val[:, 0, :, :] = x_train_val[:, 1, :, :]
        x_train_val[:, 1, :, :] = x_train_val[:, 2, :, :]
        x_train_val[:, 2, :, :] = x_train_val[:, 3, :, :]
        x_train_val[:, 3, :, :] = x_train_val[:, 4, :, :]
        x_train_val[:, 4, :, :] = x_train_val[:, 5, :, :]
        x_train_val[:, 5, :, :] = x_train_val[:, 6, :, :]
        x_train_val[:, 6, :, :] = x_train_val[:, 7, :, :]
        x_train_val[:, 7, :, :] = x_train_val[:, 8, :, :]
        x_train_val[:, 8, :, :] = x_train_val[:, 9, :, :]
        x_train_val[:, 9, :, :] = y_pred_val

        
        # Save the prediction
        prediction_array = y_pred_val.squeeze().detach().numpy()
        print(y_pred_val.squeeze().detach().numpy())
        predictions.append(prediction_array)



    predictions = np.array(predictions)

    # Calculate the RMSE for the model
    if len(x_test__[:30]) >= 30:
        rmse = np.sqrt(np.mean((predictions - x_test__[:30])**2))
        value_range = np.max(x_test__[:30]) - (-np.min(x_test__[:30]))  # Range of data: max - min

        # Calculate relative error
        relative_error = rmse / value_range

    else:
        print("The length of the test data is less than 30")
        rmse = 0

    print(f'The RMSE of the model is: {rmse}')

    #calc the relative error
    relative_error = np.linalg.norm(predictions - x_test__[:30]) / np.linalg.norm(x_test__[:30])
    #maybe do this instead of the error

    return rmse, relative_error



resolutions = [150, 180,200, 230,250,280, 300,330, 350]
#calculate the delta t for each of the resolutions
dts = []
for i in range(len(resolutions)):
    dts.append(1/resolutions[i])




rmses_model_1 = []
rmses_model_2 = []
for i in range(len(resolutions)):
    _,a = calc_the_rmse_for_model1_for_a_specific_resolution(resolutions[i])
    rmses_model_1.append(a)
    _,b = calc_the_rmse_for_model2_for_a_specific_resolution(resolutions[i])

    rmses_model_2.append(b)


#different timestep resolutions


#make a plot of the rmse for the two models
plt.figure()
plt.plot(dts, rmses_model_1, label='Model 1 ',marker='o', linestyle='-')
plt.plot(dts, rmses_model_2, label='Model 2',marker='o', linestyle='-')
plt.xlabel('Time resolution (dt)')
plt.title('Relative error of multistep prediction for different time resolutions')
plt.ylabel('Relative error')
plt.legend()
plt.grid(True)
plt.show()


#make the same plot but with log
plt.figure()
plt.plot(dts, rmses_model_1, label='Model 1',marker='o', linestyle='-')
plt.plot(dts, rmses_model_2, label='Model 2',marker='o', linestyle='-')
plt.xlabel('Time resolution (dt)')
plt.title('Relative error of multistep prediction for different time resolutions')    
plt.ylabel('Relative error')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()



###now I need to make a plot of the time taken for the two models
# Define the indices for the plots
#indices = [155, 205, 256]
#maybe do this for the best resolution
#run the model and compare the time taken

# Create a figure with subplots
#fig, axes = plt.subplots(nrows=1, ncols=len(indices), figsize=(16, 5))

# Plot for each index
#for i, index in enumerate(indices):
#    ax = axes[i]
#    ax.plot(u[index], label='Actual')
#    ax.plot(predictions[index], label='Predicted')
#    ax.legend()
#    ax.set_xlabel('Index')
#    ax.set_ylabel('Value')
#    ax.set_title(f'Time step {index}')

# Save the figure
#plt.tight_layout()
#plt.savefig('figs/subplots.png')
#plt.show()

# for M_10 modellen kan man sammenligne hvordan modellen ville performe på forskellige resolution 
# 

