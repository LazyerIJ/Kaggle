'''
https://www.kaggle.com/robikscube/nfl-big-data-bowl-plotting-player-position
'''
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_football_field(ax,
                          linenumbers=True,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=50,
                          highlighted_name='Line of Scrimmage',
                          fifty_is_los=False,
                          figsize=(12, 6.33)):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0)

    # fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    ax.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
            80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
            [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
            53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
            color='white')
    if fifty_is_los:
        ax.plot([60, 60], [0, 53.3], color='gold')
        ax.text(62, 50, '<- Player Yardline at Snap', color='gold')
    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    ax.set_xlim(0, 120)
    ax.set_ylim(-5, 58.3)
    ax.axis('off')
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            ax.text(x, 5, str(numb - 10),
                    horizontalalignment='center',
                    fontsize=20,  # fontname='Arial',
                    color='white')
            ax.text(x - 0.95, 53.3 - 5, str(numb - 10),
                    horizontalalignment='center',
                    fontsize=20,  # fontname='Arial',
                    color='white', rotation=180)
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')

    if highlight_line:
        hl = highlight_line_number
        ax.plot([hl, hl], [0, 53.3], color='yellow')
        ax.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                color='yellow')
    return ax


def loc_to_degree(file_name='stadium_loc.csv',
                  save_name='stadium_loc_degree.csv'):
    import math
    import pandas as pd
    import os
    input_dir = os.path.join('../../input')
    file_name = os.path.join(input_dir, file_name)
    save_name = os.path.join(input_dir, save_name)
    if os.path.exists(save_name):
        print('[*]File {} already exists'.format(save_name))
        return save_name
    stadium_loc = pd.read_csv(file_name, header=None)
    stadium_loc.columns = ['Stadium', 'lat1', 'lon1', 'lat2', 'lon2']

    def loc_to_degree(row):
        angle = math.atan2(row['lon2']-row['lon1'],
                           row['lat2']-row['lat1'])
        degree = math.degrees(angle)
        return abs(degree)
    stadium_loc['LocDegree'] = stadium_loc.apply(loc_to_degree, axis=1)
    stadium_loc.to_csv(save_name, index=False)
    return save_name
