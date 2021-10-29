nrows=2 #3
ncols=3 #2

# Define the figure and each axis for the row and 
fig, axs = plt.subplots(nrows=nrows,ncols=ncols,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(16.5,8)) #width, height

# axs is a 2 dimensional array of `GeoAxes`.  We will flatten it into a 1-D array
axs=axs.flatten()

xlim = [ds['lon'].values.min(), ds['lon'].values.max()]
ylim = [ds['lat'].values.min(), ds['lat'].values.max()]

xticks = np.arange(140,152,2)  #lon
yticks = np.arange(-38,-32,2)   #lat

#Loop over all of the models
for i in np.arange(6):
    data = data_list[i]
    axs[i].set_title(title_list[i])
        
#     # Add the cyclic point
#         data,lons=add_cyclic_point(data,coord=ds['lon'])

    # Contour plot
    cs=axs[i].contourf(ds['lon'],ds['lat'],data,clevs,
                          transform = ccrs.PlateCarree(),
                          cmap=cmapSel,extend='both')   #cmap options: coolwarm,

    # Draw the coastines for each subplot
    axs[i].coastlines()
    axs[i].add_feature(cfeature.OCEAN, zorder=2, edgecolor='k', facecolor='w')

    # Longitude labels
    axs[i].set_xticks(xticks, crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    axs[i].xaxis.set_major_formatter(lon_formatter)
    axs[i].set_xlim(xlim)

    # Latitude labels
    axs[i].set_yticks(yticks, crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    axs[i].yaxis.set_major_formatter(lat_formatter)
    axs[i].set_ylim(ylim)

# Delete the unwanted axes
# for i in [5]:
#     fig.delaxes(axs[i])

# Adjust the location of the subplots on the page to make room for the colorbar
fig.subplots_adjust(bottom=0.1, top=0.95, left=0.05, right=0.95,
                    wspace=0.1, hspace=0.08)

# Add a colorbar axis at the bottom of the graph
cbar_ax = fig.add_axes([0.3, 0.03, 0.4, 0.03])

# Draw the colorbar
cbar=fig.colorbar(cs, cax=cbar_ax,orientation='horizontal', label=cBarText)

# Add a big title at the top
plt.suptitle(mainTitle)
plt.savefig(outdir+figname)