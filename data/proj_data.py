"""
"""

#| - Import Modules
import os
import sys

# #########################################################
import plotly.graph_objs as go
#__|


metal_atom_symbol = "Ir"

compenv = os.environ["COMPENV"]


compenvs = ["nersc", "sherlock", "slac"]



#| - Plot Formatting

#| - Stoich. colors

stoich_color_dict = {

    # "AB2": "orange",
    # "AB3": "blue",
    "AB2": "#46cf44",
    "AB3": "#42e3e3",

    "None": "red",
    }

#__|


#| - Plotly shared stuff
scatter_shared_props = go.Scatter(
    marker=go.scatter.Marker(
        size=12,
        opacity=0.8,
        ),
    )



# shared_layout =

#| - Main layout object
layout_shared = go.Layout(

    #| - __temp__
    # angularaxis=None,
    # annotations=None,
    # annotationdefaults=None,
    # autosize=None,
    # bargap=None,
    # bargroupgap=None,
    # barmode=None,
    # barnorm=None,
    # boxgap=None,
    # boxgroupgap=None,
    # boxmode=None,
    # calendar=None,
    # clickmode=None,
    # coloraxis=None,
    # colorscale=None,
    # colorway=None,
    # datarevision=None,
    # direction=None,
    # dragmode=None,
    # editrevision=None,
    # extendfunnelareacolors=None,
    # extendpiecolors=None,
    # extendsunburstcolors=None,
    #
    # funnelareacolorway=None,
    # funnelgap=None,
    # funnelgroupgap=None,
    # funnelmode=None,
    # geo=None,
    # grid=None,
    # hiddenlabels=None,
    # hiddenlabelssrc=None,
    # hidesources=None,
    # hoverdistance=None,
    # hoverlabel=None,
    # hovermode=None,
    # images=None,
    # imagedefaults=None,
    # legend=None,
    # mapbox=None,
    #
    # scene=go.layout.Scene(
    #     annotations=None,
    #     annotationdefaults=None,
    #     aspectmode=None,
    #     aspectratio=None,
    #     bgcolor=None,
    #     camera=None,
    #     domain=None,
    #     dragmode=None,
    #     hovermode=None,
    #     uirevision=None,
    #     xaxis=None,
    #     yaxis=None,
    #     zaxis=None,
    #     ),
    #
    # selectdirection=None,
    # selectionrevision=None,
    # separators=None,
    # shapes=None,
    # shapedefaults=None,
    # showlegend=None,
    # sliders=None,
    # sliderdefaults=None,
    # spikedistance=None,
    # sunburstcolorway=None,
    # template=None,
    # ternary=None,
    # title=None,
    # titlefont=None,
    # transition=None,
    # uirevision=None,
    # updatemenus=None,
    # updatemenudefaults=None,
    # violingap=None,
    # violingroupgap=None,
    # violinmode=None,
    # waterfallgap=None,
    # waterfallgroupgap=None,
    # waterfallmode=None,
    # xaxis=None,
    # yaxis=None,

    # piecolorway=None,
    # polar=None,
    # radialaxis=None,

    # meta=None,
    # metasrc=None,
    # modebar=None,
    # orientation=None,

    #__|

    font=go.layout.Font(
        color="black",
        family="Arial",
        size=None,
        ),

    # margin=go.layout.Margin(
    #     autoexpand=None,
    #     b=None,
    #     l=None,
    #     pad=None,
    #     r=None,
    #     t=None,
    #     ),
    # #########################################################################
    paper_bgcolor="white",
    plot_bgcolor="white",

    # #########################################################################

    # #########################################################################
    # height=5.291667 * 37.795275591,
    # width=17.7 * 37.795275591,

    # height=None,
    # width=None,
    )


#| - Axis Layout options

#| - shared axis dict
font_axis_title_size = (4. / 3.) * 14

font_tick_labels_size = (4. / 3.) * 12

font_family = "Arial"

shared_axis_dict = dict(

    #| - __TEMP__
    # anchor=None,
    # automargin=None,
    # autorange=None,
    # calendar=None,
    # categoryarray=None,
    # categoryarraysrc=None,
    # categoryorder=None,
    # color=None,
    # constrain=None,
    # constraintoward=None,
    # dividercolor=None,
    # dividerwidth=None,
    # domain=None,
    # dtick=None,
    # exponentformat=None,
    # fixedrange=None,
    # gridcolor=None,
    # gridwidth=None,
    # hoverformat=None,
    # layer=None,
    # linewidth=None,
    # matches=None,
    # nticks=None,
    # overlaying=None,
    # position=None,
    # range=None,
    # rangemode=None,
    # scaleanchor=None,
    # scaleratio=None,
    # separatethousands=None,
    # showdividers=None,
    # showexponent=None,
    # showspikes=None,
    # showticklabels=None,
    # showtickprefix=None,
    # showticksuffix=None,
    # side=None,
    # spikecolor=None,
    # spikedash=None,
    # spikemode=None,
    # spikesnap=None,
    # spikethickness=None,
    # tick0=None,
    # tickangle=None,


    # go.layout.Title(
    #     arg=None,
    #     font=None,
    #     pad=None,
    #     text=None,
    #     x=None,
    #     xanchor=None,
    #     xref=None,
    #     y=None,
    #     yanchor=None,
    #     yref=None,
    #     # font=None, text=None
    #     ),
    #
    # titlefont=None,
    # type=None,
    # uirevision=None,
    # visible=None,


    # tickformat=None,
    # tickformatstops=None,
    # tickformatstopdefaults=None,
    # ticklen=None,
    # tickmode=None,  # 'auto', 'linear', 'array'
    # tickprefix=None,
    #
    #
    # tickson=None,
    # ticksuffix=None,
    # ticktext=None,
    # ticktextsrc=None,
    # tickvals=None,
    # tickvalssrc=None,
    # tickwidth=None,

    #__|

    linecolor="black",

    # #########################################################################
    mirror=True,
    showgrid=False,

    # #########################################################################
    showline=True,
    tickcolor="black",
    tickfont=dict(
        color=None,
        family=font_family,
        size=font_tick_labels_size,
        ),

    ticks="outside",  # outside", "inside", ""

    title=dict(
        font=dict(
            color="black",
            family=font_family,
            size=font_axis_title_size,
            ),
        # text="TEMP",
        ),

    # #########################################################################
    # zeroline=True,
    # zerolinecolor="#cfcfcf",
    # zerolinewidth=1.,

    )
#__|

xaxis_layout = go.layout.XAxis(shared_axis_dict)
xaxis_layout.update(go.layout.XAxis(
    title=dict(
        # text="TEMP",
        ),
    ))

yaxis_layout = go.layout.YAxis(shared_axis_dict)
yaxis_layout.update(go.layout.YAxis(
    title=dict(
        # text="TEMP",
        ),
    ))

layout_shared.xaxis = xaxis_layout
layout_shared.yaxis = yaxis_layout
#__|


#__|


#__|



#__|
