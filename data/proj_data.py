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

adsorbates = ["o", "oh", "bare", ]

#| - OER Data *********************************************
scaling_dict_mine = {
    'oh':  {'m': 1.0,  'b': 0.0},
    'o':   {'m': 2.0,  'b': 0.0},
    'ooh': {'m': 1.16, 'b': 2.8},
    }

#__|

#| - Plot Formatting **************************************

#| - Stoich. colors

stoich_color_dict = {

    # "AB2": "orange",
    # "AB3": "blue",

    # "AB2": "#46cf44",
    # "AB3": "#42e3e3",

    # "AB2": "#96272c",
    # "AB3": "#6c94b6",

    "AB2": "#42bd3e",
    "AB3": "#242424",

    "None": "red",
    }

#__|

font_axis_title_size = (4. / 3.) * 18
#  font_axis_title_size = (4. / 3.) * 28

font_tick_labels_size = (4. / 3.) * 15

font_axis_title_size__pub = (4. / 3.) * 12
font_tick_labels_size__pub = (4. / 3.) * 10


font_family = "Arial"

scatter_marker_size = 12
#  scatter_marker_size = 8

# scatter_marker_props = dict(
scatter_marker_props = go.scatter.Marker(
    size=scatter_marker_size,
    # opacity=0.8,
    symbol="circle",
    line={
        "color": "black",
        "width": 1.0,
        },
    )

#| - Plotly shared stuff
scatter_shared_props = go.Scatter(
    marker=scatter_marker_props,
    )


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
    # #####################################################
    paper_bgcolor="white",
    plot_bgcolor="white",

    # #####################################################

    # #####################################################
    # height=5.291667 * 37.795275591,
    # width=17.7 * 37.795275591,

    # height=None,
    # width=None,
    )


#| - Axis Layout options


#| - shared axis dict

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



#| - Systems to ignore

# Final systems in df_features_targets to simply ignore, likely because they are outliers in some extreme ways

sys_to_ignore__df_features_targets = [
    # Fri May 28 14:22:28 PDT 2021
    # These were removed because they had wrong octa_vol
    # The unit cells were so small, one oxygen would serve the role of more than one oxygen, if that makes sense
    ("sherlock", "hahesegu_39", 20.0, ),
    ("sherlock", "hahesegu_39", 21.0, ),

    # These were removed because their active_o_metal_dist was too large, outliers w.r.t. the rest of the data set
    ('nersc', 'hevudeku_30', 74.0),
    ('sherlock', 'wafitemi_24', 29.0),
    ('sherlock', 'kapapohe_58', 29.0),
    ('nersc', 'legofufi_61', 93.0),
    ('nersc', 'legofufi_61', 91.0),
    ('sherlock', 'ripirefu_15', 49.0),

    ('slac', 'rapebiba_65', 64.0),
    ('sherlock', 'kobehubu_94', 50.0),
    ('sherlock', 'kapapohe_58', 29.0),
    ('slac', 'fodopilu_17', 25.0),
    ('sherlock', 'wafitemi_24', 29.0),
    ('nersc', 'legofufi_61', 91.0),
    ('nersc', 'legofufi_61', 93.0),
    ('nersc', 'hevudeku_30', 74.0),
    ('sherlock', 'vipikema_98', 48.0),
    ('sherlock', 'ripirefu_15', 49.0),
    ('sherlock', 'kapapohe_58', 34.0),
    ('sherlock', 'wafitemi_24', 34.0),


    # These are systems that belong have bulk_id=cqbrnhbacg
    ('sherlock', 'kobehubu_94', 52.0),
    ('sherlock', 'kobehubu_94', 60.0),
    ('sherlock', 'vipikema_98', 47.0),
    ('sherlock', 'vipikema_98', 53.0),
    ('sherlock', 'vipikema_98', 60.0),
    ('slac', 'dotivela_46', 26.0),
    ('slac', 'dotivela_46', 32.0),
    ('slac', 'ladarane_77', 15.0),

    # Something wrong with this one, don't remember
    ("sherlock", "telibose_95", 35.0),

    # These were removed because they were duplicates
    ("slac", "gesumule_22", 32.0),
    ("slac", "fokigemi_06", 32.0),

    # These are two outlier systems, they don't have an *OH derived *O calc, 2 of 5 such systems
    # These systems also have high magmom mismatch, they are outside the 94th percentile
    ('sherlock', 'sifebelo_94', 61.0),
    ('sherlock', 'sifebelo_94', 62.0),
    ]

#__|
