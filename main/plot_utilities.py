import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# data: pd.DataFrame() with the form {'x1':{x1_sensitivity_vals...}, 'x2':{x2vals...}, ... 'y':{yvals}}
# y: y column name as str
# base_idx: index of the row that contains the base case as int (default: 0)
def spiderplot(data, y, base_idx=0):
    data_pu = data/data.iloc[base_idx].abs()
    data_pu = data_pu.reset_index(drop=True)
    columns = data.drop(y,axis=1).columns
    for c in columns:
        index_plot = ((data_pu).drop([c,y],axis=1)==1).all(axis=1).where(lambda x: x).dropna().index
        data_plot = data_pu.iloc[index_plot].sort_values(by=c)
        plt.plot(data_plot[c],data_plot[y],label=c)




def tornado_chart(labels, midpoint, low_values, high_values, title="Tornado Diagram"):
    """
    Parameters
    ----------
    labels : np.array()
        List of label titles used to identify the variables, y-axis of bar
        chart. The length of labels is used to iterate through to generate 
        the bar charts.
    midpoint : float
        Center value for bar charts to extend from. In sensitivity analysis
        this is often the 'neutral' or 'default' model output.
    low_values : np.array()
        An np.array of the model output resulting from the low variable 
        selection. Same length and order as labels. 
    high_values : np.array()
        An np.array of the model output resulting from the high variable
        selection. Same length and order as labels.
    """
    
    color_low = '#e1ceff'  # Azul claro para low_values
    color_high = '#ff6262' # Rojo para high_values
    
    ys = range(len(labels))  
    
    fig, ax = plt.subplots(figsize=(9, len(labels) * 0.5 + 4))
    
    for y, low_value, high_value in zip(ys, low_values, high_values):
    
        low_width = midpoint - low_value
        high_width = high_value - midpoint
    
        ax.broken_barh(
            [
                (low_value, low_width),
                (midpoint, high_width)
            ],
            (y - 0.4, 0.5),  
            facecolors=[color_low, color_high],
            edgecolors=['black', 'black'],
            linewidth=0.5
        )
        
        offset_x = 0.5
        offset_y = 0.2

        ax.text(low_value - offset_x, y + offset_y, f"{low_value:.2f}", va='center', ha='right', fontsize=10, fontweight='bold')
        ax.text(high_value + offset_x, y + offset_y, f"{high_value:.2f}", va='center', ha='left', fontsize=10, fontweight='bold')
    
    ax.axvline(midpoint, color='black', linewidth=1)

    ax.spines[['right', 'left', 'top']].set_visible(False)
    ax.set_yticks([])
    
    ax.text(midpoint, len(labels)-0.4, title, color='black', fontsize=15, va='center', ha='center', fontweight='bold')

    ax.set_xlabel('+y, -y')
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)

    x_min = min(low_values) - 5 
    x_max = max(high_values) + 5  
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, len(labels) - 0.5)
    
    ax.tick_params(left=False)

    low_patch = mpatches.Patch(color=color_low, label='- y')
    high_patch = mpatches.Patch(color=color_high, label='+ y')
    ax.legend(handles=[low_patch, high_patch], loc='upper right', fontsize=10, frameon=False)

    plt.show()

    return

tornado_chart(labels, midpoint, low_values, high_values)