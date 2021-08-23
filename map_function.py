#!/usr/bin/env python
# coding: utf-8

# In[1]:


# DR LI WAN | UNIVERSITY OF CAMBRIDGE
# MR SHANTONG WANG
# MS TIANYUAN WANG

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import matplotlib.lines as mlines
import matplotlib.patheffects as path_effects
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import cm

import plotly.express as px
import networkx as nx
from pyproj import Proj, transform 
from mpl_toolkits.basemap import Basemap as Basemap

import geopandas as gpd  # gpd.__version__  # gpd.show_versions()
import shapely
import shapely.geometry as geom
from shapely.geometry import LineString


# In[2]:


def read_csv(filename):
    
    global df
    
    df = pd.read_csv(filename)
    #df = pd.read_csv('with nodes coordinates.csv') #X: longitude, Y:latitude
    # df = df.rename(columns = {'X':'long', 'Y':'lat'})
    # print(df.shape)
    df.head()

    epsg_in = 'epsg:3857'
    epsg_out = 'epsg:4326'
    inProj = Proj(init = epsg_in) 
    outProj = Proj(init= epsg_out)
    df['long'], df['lat'] = transform(inProj, outProj, df['x'].tolist(), df['y'].tolist())

#     df = df.rename(columns = {'X':'long',
#                               'Y':'lat'})

    df = df.round({'Shape_Leng': 5, 'distance': 5}) #otherwise, distance can't fully == Shape_Leng
    #df
    
    # x1y1 is the start point, x2y2 is the end point
    x1y1 = df.loc[df['distance']==0][['OBJECTID', 'Shape_Leng','GroupCount','long', 'lat']]
    x1y1 = x1y1.rename(columns = {'long':'coord_x1', 'lat':'coord_y1'})
    x1y1['name_x1y1'] = x1y1['OBJECTID'].astype(str) + 'a'

    x2y2 = df.loc[df['distance']==df['Shape_Leng']][['OBJECTID', 'Shape_Leng','GroupCount','long', 'lat']]
    x2y2 = x2y2.rename(columns = {'long':'coord_x2', 'lat':'coord_y2'})
    x2y2['name_x2y2'] = x2y2['OBJECTID'].astype(str) + 'b'

    df = pd.merge(x1y1, x2y2, on=['OBJECTID', 'Shape_Leng','GroupCount'])
    df = df.rename(columns = {'GroupCount':'magnitude',
                              'Shape_Leng':'length_unit'})
    #print(df.shape)
    
    return df 
    


# In[3]:


def read_shp(filename):
    
    global df1
    
    df1 = gpd.read_file(filename)
    
    return df1


# In[4]:


def flow_map(file,title,datetime,vehicle,location,legend_title):
    
#              cutoff1,cutoff2,cutoff3,cutoff4,cutoff5,
#              color1,color2,color3,color4,color5

#     inProj = Proj(init = epsg_in) 
#     outProj = Proj(init= epsg_out)
#     df['long_x1y1'], df['lat_x1y1'] = transform(inProj, outProj, df[coord_x1].tolist(), df[coord_y1].tolist())
#     df['long_x2y2'], df['lat_x2y2'] = transform(inProj, outProj, df[coord_x2].tolist(), df[coord_y2].tolist())
 
#     df = df[df[magnitude] > minimumvalue]

    subtitle = location+' '+datetime+' '+vehicle

    cutoff1 = 0
    cutoff2 = 1
    cutoff3 = 100
    cutoff4 = 2000
    cutoff5 = 5000
    color1 = 'lightgreen'
    color2 = 'dodgerblue'
    color3 = 'khaki'
    color4 = 'orange'
    color5 = 'orangered'

    annotate1 = 'Cambridge'
    dpi = 300  
    minimumvalue = 0

    magnitude = 'magnitude'
    name1='name_x1y1'
    coord_x1='coord_x1'
    coord_y1='coord_y1'
    name2='name_x2y2'
    coord_x2='coord_x2'
    coord_y2='coord_y2'
    service = 'Ocean_Basemap'
    epsg = 4326
    xpixels = 1500
    epsg_in = 'epsg:3857'
    epsg_out = 'epsg:4326'
    
    
#     graph = nx.from_pandas_edgelist(df, name1, name2, edge_attr=magnitude)
    graph1 = nx.from_pandas_edgelist(df1, 'Name', 'Name')    
        
    # llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon are the lat/lon values of the lower left and upper right corners of the map.
    fig = plt.figure(figsize = (9,7), dpi = dpi)
    fig.patch.set_facecolor('#f5f5f5') ##e9e9e9 #black
    ax = fig.add_subplot(1,1,1)
    
    m = Basemap(projection='merc', resolution='i', #epsg=epsg,
                llcrnrlon=df1['X'].min()-0.025, 
                llcrnrlat=df1['Y'].min()-0.005, 
                urcrnrlon=df1['X'].max()+0.025, 
                urcrnrlat=df1['Y'].max()+0.005,
                lat_ts=df1['Y'].mean(),
                suppress_ticks=True)
   
    m.readshapefile(r'cambridge planning/roadnew_epsg4326', 'cambridge', drawbounds=True, linewidth=0.3, color='#4C4C4C', zorder=0)
    m.readshapefile(r'gadm36_GBR_shp/gadm36_GBR_3', 'states', drawbounds=True, linewidth=0.3, color='#4C4C4C', zorder=0)
    m.readshapefile(file, 'roads', drawbounds=True, linewidth=2, color='#9eccf0', zorder=0)
    #m.readshapefile(r'new1/anpr_epsg4326', 'states', drawbounds=True, linewidth=2, color='#cbeba0', zorder=0)
    #m.arcgisimage(service=service, xpixels = 1500, verbose= False)
    
    road_names = []
    for shape_dict in m.roads_info:
        road_names.append(shape_dict['OBJECTID'])
    
    for info, shape in zip(m.roads_info, m.roads):
        if info['GroupCount']!=None:
            if (info['GroupCount']>=cutoff1)&(info['GroupCount']<cutoff2):
                x, y = zip(*shape)
                m.plot(x, y, marker=None,color=color1,label = '%.0f - %.0f'%(cutoff1,cutoff2))

            elif (info['GroupCount']>=cutoff2)&(info['GroupCount']<cutoff3):
                x, y = zip(*shape)
                m.plot(x, y, marker=None,color=color2,label = '%.0f - %.0f'%(cutoff2,cutoff3))

            elif (info['GroupCount']>=cutoff3)&(info['GroupCount']<cutoff4):
                x, y = zip(*shape)
                m.plot(x, y, marker=None,color=color3,label = '%.0f - %.0f'%(cutoff3,cutoff4))

            elif (info['GroupCount']>=cutoff4)&(info['GroupCount']<cutoff5):
                x, y = zip(*shape)
                m.plot(x, y, marker=None,color=color4,label = '%.0f - %.0f'%(cutoff4,cutoff5))

            elif info['GroupCount']>=cutoff5:
                x, y = zip(*shape)
                m.plot(x, y, marker=None,color=color5,label = '> %.0f'%(cutoff5))
    
    legend1 = mpatches.Patch(color=color1, label='%.0f - %.0f'%(cutoff1,cutoff2))
    legend2 = mpatches.Patch(color=color2, label='%.0f - %.0f'%(cutoff2,cutoff3))
    legend3 = mpatches.Patch(color=color3, label='%.0f - %.0f'%(cutoff3,cutoff4))
    legend4 = mpatches.Patch(color=color4, label='%.0f - %.0f'%(cutoff4,cutoff5))
    legend5 = mpatches.Patch(color=color5, label='> %.0f'%(cutoff5))
    plt.legend(handles=[legend1,legend2,legend3,legend4,legend5], title=legend_title,#fontsize=6, #title_fontsize=6.5,
               loc = 'lower left',prop={'size':7},handleheight=0.5) #markerscale=0.9,
    
    
    mx1, my1 = m(df1['X'].values, df1['Y'].values)
    pos1 = {}
    for count, elem in enumerate (df1['Name']):
        pos1[elem] = (mx1[count], my1[count])    
    
    nx.draw_networkx_nodes(G = graph1, pos = pos1, nodelist = graph1.nodes(),
                            node_shape='o', node_color = '#ababab',   #898c8f #c0d0e1
                            alpha = 0.5, node_size =20, linewidths=0.75, edgecolors = '#1c1c1c')
                            
    
    m.drawcountries(linewidth = 0.3)
    #m.drawcoastlines(linewidth=0.5)      #not work in Binder - but no difference, just dropped this line
    m.fillcontinents(alpha = 0.05,zorder=0)

    # text = plt.annotate(annotate1, m((df_name['long'].mean()),(df_name['lat'].mean())), color='#232323', 
    #                     fontsize=10, fontname='Arial', ha='center', va='center') 
    # text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='whitesmoke'), path_effects.Normal()])
    
    ax.text(0, 1.05, title , transform=ax.transAxes, size=13, weight=600, ha='left', color = 'gray')
    ax.text(0, 1.02, subtitle , transform=ax.transAxes, size=7, color='gray') #weight=600, ha='left')'lightgray'

    #plt.savefig("./flow map.jpeg", format = "jpeg", dpi = 300)
    ax.axis('off')
    plt.show()


# In[5]:


def flow_map1(title,datetime,vehicle,location,legend_title):
    
#              cutoff1,cutoff2,cutoff3,cutoff4,cutoff5,
#              color1,color2,color3,color4,color5
    
#     global df

#     inProj = Proj(init = epsg_in) 
#     outProj = Proj(init= epsg_out)
#     df['long_x1y1'], df['lat_x1y1'] = transform(inProj, outProj, df[coord_x1].tolist(), df[coord_y1].tolist())
#     df['long_x2y2'], df['lat_x2y2'] = transform(inProj, outProj, df[coord_x2].tolist(), df[coord_y2].tolist())
 
#     df = df[df[magnitude] > minimumvalue]

#     title = 'Cambridge Flow Map'
#     subtitle = 'No. of commuters by mode - xxx in year xxx'

    subtitle = location+' '+datetime+' '+vehicle

    cutoff1 = 0
    cutoff2 = 1
    cutoff3 = 100
    cutoff4 = 2000
    cutoff5 = 5000
    color1 = 'lightgreen'
    color2 = 'dodgerblue'
    color3 = 'khaki'
    color4 = 'orange'
    color5 = 'orangered'

    annotate1 = 'Cambridge'
    dpi = 300
    minimumvalue = 0

    magnitude = 'magnitude'
    name1='name_x1y1'
    coord_x1='coord_x1'
    coord_y1='coord_y1'
    name2='name_x2y2'
    coord_x2='coord_x2'
    coord_y2='coord_y2'
    service = 'Ocean_Basemap'
    epsg = 4326
    xpixels = 1500
    epsg_in = 'epsg:3857'
    epsg_out = 'epsg:4326'
    
    
    
    graph = nx.from_pandas_edgelist(df, name1, name2, edge_attr=magnitude)
    graph1 = nx.from_pandas_edgelist(df1, 'Name', 'Name')

    df_name1 = df[[name1,'coord_x1','coord_y1']].rename(columns={name1:'name', 'coord_x1':'long', 'coord_y1':'lat'}).drop_duplicates()
    df_name2 = df[[name2,'coord_x2','coord_y2']].rename(columns={name2:'name', 'coord_x2':'long', 'coord_y2':'lat'}).drop_duplicates()
    df_name = pd.concat([df_name1, df_name2], ignore_index=True, sort=False).drop_duplicates()    
    
        
    # llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon are the lat/lon values of the lower left and upper right corners of the map.
    fig = plt.figure(figsize = (9,7), dpi = dpi)
    fig.patch.set_facecolor('#f5f5f5') ##e9e9e9 #black
    ax = fig.add_subplot(1,1,1)
    
    m = Basemap(projection='merc', resolution='i', #epsg=epsg,
            llcrnrlon=df_name['long'].min()-0.04, 
            llcrnrlat=df_name['lat'].min()-0.005, 
            urcrnrlon=df_name['long'].max()+0.04, 
            urcrnrlat=df_name['lat'].max()+0.005,
            lat_ts=df_name['lat'].mean(),
            suppress_ticks=True)
   
    m.readshapefile(r'cambridge planning/roadnew_epsg4326', 'roads', drawbounds=True, linewidth=0.3, color='#4C4C4C', zorder=0)
    m.readshapefile(r'gadm36_GBR_shp/gadm36_GBR_3', 'states', drawbounds=True, linewidth=0.3, color='#4C4C4C', zorder=0)
    #m.arcgisimage(service=service, xpixels = 1500, verbose= False)


    mx, my = m(df_name['long'].values, df_name['lat'].values)
    pos = {}
    for count, elem in enumerate (df_name['name']):
        pos[elem] = (mx[count], my[count])
    
    mx1, my1 = m(df1['X'].values, df1['Y'].values)
    pos1 = {}
    for count, elem in enumerate (df1['Name']):
        pos1[elem] = (mx1[count], my1[count])
        

    
    durations = [np.log(i[magnitude]*0.01) for i in dict(graph.edges).values()]
    
    
    nx.draw_networkx_nodes(G = graph1, pos = pos1, nodelist = graph1.nodes(),
                            node_shape='o', node_color = '#ababab',   #898c8f #c0d0e1
                            alpha = 0.5, node_size =12, linewidths=0.5, edgecolors = '#151515')
    

    nx.draw_networkx_nodes(G = graph, pos = pos, nodelist = graph.nodes(),
                            node_shape='o', node_color = '#898c8f',   #c0d0e1
                            alpha = 0.5, node_size =0)

    nx.draw_networkx_edges(G = graph, pos = pos, edgelist = [edge for edge in graph.edges(data=True) if (edge[2][magnitude] >= cutoff1)&(edge[2][magnitude] < cutoff2)],
                            edge_color = color1, label = '%.0f - %.0f'%(cutoff1,cutoff2),
                            width=2, #width=durations,
                            alpha=0.35, arrows = False)

    nx.draw_networkx_edges(G = graph, pos = pos, edgelist = [edge for edge in graph.edges(data=True) if (edge[2][magnitude] >= cutoff2)&(edge[2][magnitude] < cutoff3)],
                            edge_color = color2, label= '%.0f - %.0f'%(cutoff2,cutoff3),
                            width=2, #width=durations,
                            alpha=0.35, arrows = False)

    nx.draw_networkx_edges(G = graph, pos = pos, edgelist = [edge for edge in graph.edges(data=True) if (edge[2][magnitude] >= cutoff3)&(edge[2][magnitude] < cutoff4)],
                            edge_color = color3, label= '%.0f - %.0f'%(cutoff3,cutoff4),
                            width=2, #width=durations,
                            alpha=0.5, arrows = False)

    nx.draw_networkx_edges(G = graph, pos = pos, edgelist = [edge for edge in graph.edges(data=True) if (edge[2][magnitude] >= cutoff4)&(edge[2][magnitude] < cutoff5)],
                            edge_color = color4, label= '%.0f - %.0f'%(cutoff4,cutoff5),
                            width=2, #width=durations,
                            alpha=0.75, arrows = False)

    nx.draw_networkx_edges(G = graph, pos = pos, edgelist = [edge for edge in graph.edges(data=True) if edge[2][magnitude] >= cutoff5],
                            edge_color = color5, label= '> %.0f'%(cutoff5),
                            width=2, #width=durations,
                            alpha=0.9, arrows = False)
                            
    
    m.drawcountries(linewidth = 0.3)
    #m.drawcoastlines(linewidth=0.5)      #not work in Binder - but no difference, just dropped this line
    m.fillcontinents(alpha = 0.05,zorder=0)

    plt.legend(loc = 'lower right',prop={'size':7}, title=legend_title) #,title_fontsize=7.5

    # text = plt.annotate(annotate1, m((df_name['long'].mean()),(df_name['lat'].mean())), color='#232323', 
    #                     fontsize=10, fontname='Arial', ha='center', va='center') 
    # text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='whitesmoke'), path_effects.Normal()])

    
    ax.text(0, 1.05, title , transform=ax.transAxes, size=13, weight=600, ha='left', color = 'gray')
    ax.text(0, 1.02, subtitle , transform=ax.transAxes, size=7, color='gray') #weight=600, ha='left')'lightgray'

    #plt.savefig("./flow map.jpeg", format = "jpeg", dpi = 300)
    ax.axis('off')
    plt.show()



# In[6]:


# name1='name_x1y1'
# name2='name_x2y2'
# df_name1 = df[[name1,'coord_x1','coord_y1']].rename(columns={name1:'name', 'coord_x1':'long', 'coord_y1':'lat'}).drop_duplicates()
# df_name2 = df[[name2,'coord_x2','coord_y2']].rename(columns={name2:'name', 'coord_x2':'long', 'coord_y2':'lat'}).drop_duplicates()
# df_name = pd.concat([df_name1, df_name2], ignore_index=True, sort=False).drop_duplicates()

# m = Basemap(projection='merc', resolution='i', #epsg=epsg,
# llcrnrlon=df_name['long'].min()-0.08, 
# llcrnrlat=df_name['lat'].min()-0.01, 
# urcrnrlon=df_name['long'].max()+0.08, 
# urcrnrlat=df_name['lat'].max()+0.01,
# lat_ts=df_name['lat'].mean(),
# suppress_ticks=True)

# # m.readshapefile(r'new1/anpr_epsg4326', 'states', drawbounds=True, linewidth=0.3, color='#4C4C4C', zorder=0)
# m.readshapefile(r'gadm36_GBR_shp/gadm36_GBR_3', 'states', drawbounds=True, linewidth=0.3, color='#4C4C4C', zorder=0)
# m.readshapefile(r'cambridge planning/roadnew_epsg4326', 'states', drawbounds=True, linewidth=0.3, color='#4C4C4C', zorder=0)
# m.readshapefile(r'new1/test0820_epsg4326', 'roads', drawbounds=True, linewidth=0.3, color='#4C4C4C', zorder=0)


# In[ ]:




