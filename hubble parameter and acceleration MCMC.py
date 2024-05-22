# %% [markdown]
# ### libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ### data extraction from the csv file:
# 

# %%
df = pd.read_csv('DistMod.csv')
zcmb = df['zcmb']
mb = df['mb'] #apparent magnitude
dmb = df['dmb'] #apparent magnitude error

# %% [markdown]
# ### input values 

# %%
H0 = 70
q0 = 0 
M = -19.27

# %% [markdown]
# ### define curve

# %%
def curve(q0, H0, z):
    m = M + 43.23-(5*np.log10(H0 / 68)) + 5*np.log10(z) + 1.086*(1-q0)*z
    return m

# %% [markdown]
# ### scatter plot:

# %%
plt.scatter(zcmb , mb , s = 5)
plt.title("m-z scatter plot")
plt.ylabel("m (apparent magnitude)")
plt.xlabel("z (redshift)")
plt.gca().invert_xaxis()
plt.errorbar(zcmb, mb,yerr = dmb, fmt='none',ecolor = 'black',color='yellow') 
plt.show()

# %% [markdown]
# fit 1:
# H0 = 0
# q0 = 0

# %%
q0 = -0.5
H0 = 70
plt.scatter(zcmb , mb , s = 5)
x_curve = zcmb
y_curve = curve(q0,H0,zcmb)
plt.gca().invert_xaxis()
plt.plot(x_curve,y_curve,color = 'black')
plt.show()

# %% [markdown]
# # A)
# ### with trial and error we can find out that:
# ## -1<q0<1
# 
# and 
# 
# ## 65<H0<75

# %% [markdown]
# ### penalty function

# %%
def penalty(m , z , H0, q0, dm):
    chi = (m - curve(q0 , H0, z))/dm
    return (chi)**2

# %% [markdown]
# ## phase space

# %%
'''
# Create x and y values
q0_ax = np.linspace(-5, 5, 100)
H0_ax = np.linspace(60, 90, 100)

chi2_array = np.empty((100,100))
for i in range(0 ,100):
   for j in range(0 ,100):
        q = q0_ax[i]
        h = H0_ax[j]
        total = 0
        for k in range(0,len(mb)):
            total += penalty(mb[k],zcmb[k],h,q,dmb[k])
        chi2_array[i][j] = total

plt.imshow(chi2_array , origin = 'lower')
plt.colorbar()
plt.show()


X, Y = np.meshgrid(q0_ax, H0_ax)
Z = chi2_array
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
fig.colorbar(surf)
plt.show()
'''


# %% [markdown]
# # MCMC:

# %%
import random
#find random starting point in phase space 
point_q = np.random.uniform(-2, 2)
point_h = np.random.uniform(60,80)
start = (point_q , point_h)

nmcmc = 50000
q_sigma = 0.08
h_sigma = 0.45

#point1:
chi2_1 = 0
for k in range(0,len(mb)):
    chi2_1 += penalty(mb[k],zcmb[k],point_h,point_q,dmb[k])

w_list = []
chi2_1_list  = []
point_q_list = []
point_h_list = []
acceptance = 0 
points = []

w = 1
for i in range(0,nmcmc):
    
    q_step = np.random.normal(loc=0 , scale=q_sigma)
    h_step = np.random.normal(loc=0 , scale=h_sigma)
    
    new_q = point_q + q_step
    new_h = point_h + h_step
    
    chi2_2 = 0
    for k in range(0,len(mb)):
        chi2_2 += penalty(mb[k],zcmb[k],new_h,new_q,dmb[k])

    r = random.random()

    if r < np.exp(chi2_1 - chi2_2):
            x=np.zeros((1,3))
            chi2_1_list.append(chi2_2)
            chi2_1 = chi2_2
            point_q_list.append(new_q)
            x[0,0] = new_q
            point_q = new_q
            point_h_list.append(new_h)
            x[0,1] = new_h
            point_h = new_h
            acceptance += 1
            w_list.append(w)
            x[0,2]=w
            points.append(x)
            w=1
    else:
         w += 1
         pass
acceptance_rate = acceptance/nmcmc
print("acceptance rate =",acceptance_rate*100 )

plt.scatter(point_h_list,point_q_list, s = 1)
plt.xlabel("H0")
plt.ylabel("q0")


# %%
print(points)
#(q,h,w)

# %% [markdown]
# ### eliminating the burn-in phase:

# %%
# Calculate the index corresponding to 10% of the array length
q_burn_index = int(len(point_q_list) * 0.1)
h_burn_index = int(len(point_h_list) * 0.1)
point_burn_index = int(len(points)*0.1)
# Delete the first 10% of values
new_q_list= point_q_list[q_burn_index:]
new_h_list= point_h_list[h_burn_index:]
new_point_list = points[point_burn_index:]

plt.scatter(new_h_list,new_q_list, s = 1)
plt.xlabel("H0")
plt.ylabel("q0")

# %% [markdown]
# ### creating a grid, calculating the weight array 

# %%
q_min = np.min(new_q_list)
q_max = np.max(new_q_list)
h_min = np.min(new_h_list)
h_max = np.max(new_h_list)

boxes = []
l_q = 0.01
l_h = 0.05

for i in range(0 , len(new_point_list)):
    box = np.zeros((1,3))
    lower_limit_q =  float(new_point_list[i][0,0]//l_q)
    box[0,0] = lower_limit_q
    lower_limit_h = float(new_point_list[i][0,1]//l_h)
    box[0,1] = lower_limit_h
    box[0,2] = new_point_list[i][0,2]
    boxes.append(box)

grid_numbers_q = np.arange(q_min, q_max, l_q )//l_q
grid_numbers_h = np.arange(h_min, h_max, l_h )//l_h

array = np.zeros((len(grid_numbers_q),len(grid_numbers_h)))

for i in range(len(grid_numbers_q)):
    for j in range(len(grid_numbers_h)):
        x = grid_numbers_q[i]
        y = grid_numbers_h[j]
        box_counter = 0
        for point_idx in range(len(new_point_list)):  
                box = boxes[point_idx]
                if box[0,0] == x and box[0,1] == y:
                    box_counter+=box[0,2]
                else:
                    pass
        array[i, j] = box_counter

# %% [markdown]
# ### visualising the weight array in 2D

# %%
plt.imshow(array,origin = "lower")
x_ax = np.linspace(q_min,q_max,5)
y_ax = np.linspace(h_min,h_max,5)
x_ticks = np.linspace(0,44,5)
y_ticks = np.linspace(0,48,5)
x_tick_labels = np.round(x_ax,3)
y_tick_labels = np.round(y_ax,3)
plt.xticks(x_ticks, x_tick_labels)
plt.yticks(y_ticks,y_tick_labels)

# %%
print(np.shape(array))

# %% [markdown]
# ### visualising the weight array in 3D

# %%
x = np.linspace(q_min,q_max,43)
y = np.linspace(h_min,h_max,49)
X,Y = np.meshgrid(x,y)
Z = array 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
plt.show()

# %% [markdown]
# ### choosing multiple starting points

# %%
import random
#find random starting point in phase space 
point_q_1 = np.random.uniform(-2, 2)
point_h_1 = np.random.uniform(60,80)
start_1  = (point_q_1 , point_h_1)
start 
point_q_2 = np.random.uniform(-2, 2)
point_h_2 = np.random.uniform(60,80)
start_2  = (point_q_2 , point_h_2)

point_q_3 = np.random.uniform(-2, 2)
point_h_3 = np.random.uniform(60,80)
start_3  = (point_q_3 , point_h_3)

point_q_4 = np.random.uniform(-2, 2)
point_h_4 = np.random.uniform(60,80)
start_4  = (point_q_4 , point_h_4)

nmcmc = 50000
q_sigma = 0.08
h_sigma = 0.45

starts = [start_1,start_2,start_3,start_4]
acceptance_rates = []
point_q_list_of_lists = []
point_h_list_of_lists = []

#point1:
for start in starts:
    point_q = start[0]
    point_h = start[1]
    chi2_1 = 0
    for k in range(0,len(mb)):
        chi2_1 += penalty(mb[k],zcmb[k],point_h,point_q,dmb[k])

    w_list = []
    chi2_1_list  = []
    point_q_list = []
    point_h_list = []
    acceptance = 0 
    points = []

    w = 1
    for i in range(0,nmcmc):
    
        q_step = np.random.normal(loc=0 , scale=q_sigma)
        h_step = np.random.normal(loc=0 , scale=h_sigma)
    
        new_q = point_q + q_step
        new_h = point_h + h_step
    
        chi2_2 = 0
        for k in range(0,len(mb)):
            chi2_2 += penalty(mb[k],zcmb[k],new_h,new_q,dmb[k])

        r = random.random()

        if r < np.exp(chi2_1 - chi2_2):
                x=np.zeros((1,3))
                chi2_1_list.append(chi2_2)
                chi2_1 = chi2_2
                point_q_list.append(new_q)
                x[0,0] = new_q
                point_q = new_q
                point_h_list.append(new_h)
                x[0,1] = new_h
                point_h = new_h
                acceptance += 1
                w_list.append(w)
                x[0,2]=w
                points.append(x)
                w=1
        else:
            w += 1
            pass
    acceptance_rate = acceptance/nmcmc
    acceptance_rates.append(acceptance_rate)
    point_q_list_of_lists.append(point_q_list)
    point_h_list_of_lists.append(point_h_list)


    

for i in range(0,len(point_h_list_of_lists)):
     x = point_q_list_of_lists[i]
     y = point_h_list_of_lists[i]
     plt.scatter(x,y,s=1)
plt.show()

# %%
mean_acceptance_rate = np.mean(acceptance_rates)
print("mean acceptance rate =" , mean_acceptance_rate)


