import os
import torch
import numpy as np
from scipy.stats import mode
from scipy.interpolate import griddata
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm
from typing import List, Optional, Tuple, Union
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--save_folder", type=str, default=None)
FLAGS = parser.parse_args()

def get_cmap(input=None, num=-1) :
    if input is None :
        raise RuntimeError("Can't generate a cmap without any inputs")
        
    c = sns.color_palette(input)
    
    if num != -1 :
        c = c*num
        c = c[:num]
        return matplotlib.colors.ListedColormap(c)
    else :
        return matplotlib.colors.LinearSegmentedColormap.from_list("custom", c)
    
# %%
def plot_perturb_plt(rx, ry, loss, predict,
                     z_by_loss=True, color_by_loss=False, color='viridis', 
                     min_value=None, max_value=None,
                     title=None, width=8, height=7, linewidth = 0.1,
                     x_ratio=1, y_ratio=1, z_ratio=1,
                     edge_color='#f2fafb', colorbar_yticklabels=None,
                     pane_color=(1.0, 1.0, 1.0, 0.0),
                     tick_pad_x=0, tick_pad_y=0, tick_pad_z=1.5,
                     xticks=[0.0,0.4,0.8,1.2,1.6,2.0], yticks=[0.0,0.4,0.8,1.2,1.6,2.0], zticks=None,
                     xlabel=None, ylabel=None, zlabel=None,
                     xlabel_rotation=16, ylabel_rotation=-35, zlabel_rotation=0,
                     view_azimuth=230, view_altitude=30,
                     light_azimuth=315, light_altitude=45, light_exag=0,EOT_Track_3D=None, Exact_Track_3D=None) :
    
    if z_by_loss :
        zs = loss
    else :
        zs = predict
    
    if color_by_loss :
        colors = loss
    else :
        colors = predict
    
    xs, ys = np.meshgrid(rx, ry)
    
    fig = plt.figure(figsize=(width, height))
    ax = plt.axes(projection='3d')

    if title is not None :
        ax.set_title(title)
    
    if min_value is None :
        min_value = int(colors.min())
    if max_value is None :
        max_value = int(colors.max())
        
    if 'float' in str(colors.dtype):
         scamap = cm.ScalarMappable(cmap=get_cmap(color))
    else:    
         scamap = cm.ScalarMappable(cmap=get_cmap(color, max_value-min_value+1))
   
    scamap.set_array(colors)
    scamap.set_clim(vmax=max_value+.5, vmin=min_value-.5)

    ls = LightSource(azdeg=light_azimuth, altdeg=light_altitude)
    fcolors = ls.shade(colors, cmap=scamap.cmap, vert_exag=light_exag, blend_mode='soft')
    surf = ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, facecolors=fcolors,
                           linewidth=linewidth, antialiased=True, shade=False, alpha=0.4,zorder=0.5)

    surf.set_edgecolor(edge_color)

    ax.view_init(azim=view_azimuth, elev=view_altitude)
    ########################################################
    points = np.array(EOT_Track_3D)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=40, color='orange', edgecolors='blue', zorder=5)
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1] - points[i]
        ax.quiver(start[0], start[1], start[2], end[0], end[1], end[2],
                arrow_length_ratio=0.02, color='blue', zorder=5,alpha=1, linewidth=4)

    points = np.array(Exact_Track_3D)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=40, color='green', edgecolors='red',zorder=5)

    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1] - points[i]
        ax.quiver(start[0], start[1], start[2], end[0], end[1], end[2],
                arrow_length_ratio=0.02, color='red',zorder=5,alpha=1, linewidth=4)
    ###############################################################
    # You can change 0.01 to adjust the distance between the main image and the colorbar.
    # You can change 0.02 to adjust the width of the colorbar.
    cax = fig.add_axes([ax.get_position().x1+0.01,
                        ax.get_position().y0+ax.get_position().height/8,
                        0.02,
                        ax.get_position().height/4*3])
    
    cbar = plt.colorbar(scamap,
                        ticks=np.linspace(min_value, max_value, max_value-min_value+1),
                        cax=cax)
    cbar.solids.set(alpha=0.4)
    # if colorbar_yticklabels is not None :
    #     cbar.ax.set_yticklabels(colorbar_yticklabels)
    
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    
    if xlabel is not None :
        ax.set_xlabel(xlabel, rotation=xlabel_rotation)
    if ylabel is not None :
        ax.set_ylabel(ylabel, rotation=ylabel_rotation)
    if zlabel is not None :
        ax.set_zlabel(zlabel, rotation=zlabel_rotation)
        
    if xticks is not None :
        ax.set_xticks(xticks)
    if yticks is not None :
        ax.set_yticks(yticks)
    if zticks is not None :
        ax.set_zticks(zticks)
    
    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)
    
    ax.tick_params(axis='x', pad=tick_pad_x)
    ax.tick_params(axis='y', pad=tick_pad_y)
    ax.tick_params(axis='z', pad=tick_pad_z)
    ax.set_zlim(0,16)    

savedir=FLAGS.save_folder

range_x=(0,2)
range_y=(0,2)
grid_size=30
rx = np.linspace(*range_x, grid_size)
ry = np.linspace(*range_y, grid_size)


loss_list = torch.load(f'./{savedir}/loss_list.pth', map_location='cpu')
pre_list = torch.load(f'./{savedir}/pre_list.pth', map_location='cpu')
EOT_Track_3D = torch.load(f'./{savedir}/eot_track_3d.pth', map_location='cpu')
Exact_Track_3D = torch.load( f'./{savedir}/exact_track_3d.pth', map_location='cpu')
loss_list = loss_list #- 1
EOT_Track = []
EOT1_Track = []
Exact_Track = []
for x,y,z in EOT_Track_3D:
    EOT_Track.append((x,y))
for x,y,z in Exact_Track_3D:
    Exact_Track.append((x,y))

X, Y = np.meshgrid(rx, ry)
grid_x = X.ravel()
grid_y = Y.ravel()
grid_z = loss_list.ravel() 
EOT_Track_Z = griddata((grid_x, grid_y), grid_z, np.array(EOT_Track), method='cubic')
EOT1_Track_Z = griddata((grid_x, grid_y), grid_z, np.array(EOT1_Track), method='cubic')
Exact_Track_Z = griddata((grid_x, grid_y), grid_z, np.array(Exact_Track), method='cubic')
EOT_Track_3D = []
for i, ((x,y),z) in enumerate(zip(EOT_Track, EOT_Track_Z)):
    EOT_Track_3D.append((x,y,z))
Exact_Track_3D = []
for i, ((x,y),z) in enumerate(zip(Exact_Track, Exact_Track_Z)):
    Exact_Track_3D.append((x,y,z))



zs = loss_list
colors = pre_list
if len(set(colors.reshape(-1).tolist()))==1:
    colorbar_yticklabels = ["True"]
else:
    colorbar_yticklabels = ["False", "True"]

plot_perturb_plt(rx, ry, zs, colors,
                 z_by_loss=True, color_by_loss=False,
                 color=["#6858ab", "#53cddb"],
                 min_value=None, max_value=None,
                 title=None, width=8, height=7, linewidth = 0.1,
                 x_ratio=1, y_ratio=1, z_ratio=1,
                 edge_color='#f2fafb',
                 colorbar_yticklabels=colorbar_yticklabels,
                 pane_color=(1.0, 1.0, 1.0, 0.0),
                 tick_pad_x=0, tick_pad_y=0, tick_pad_z=1.5,
                #  xticks=None, yticks=None, zticks=None,
                 xlabel=r'$Deterministc\;Attack\;Direction$', ylabel=r'$Residual\;Direction$', zlabel=r'$Loss$',
                 view_azimuth=235, view_altitude=20,
                 light_azimuth=0, light_altitude=20, light_exag=0,EOT_Track_3D=EOT_Track_3D, Exact_Track_3D=Exact_Track_3D)
plt.savefig('figure_plot.png')   
plt.savefig('figure_plot.pdf')         
           