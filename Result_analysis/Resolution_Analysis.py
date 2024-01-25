import os
import time
import pandas as pd
from Result_analysis.ResultFrame import ResultFrame


folder = "resolution_comparison"
base_path = os.getcwd() + "/../results/"
frames_mean = pd.DataFrame()
frames_std = pd.DataFrame()
frames_pos = pd.DataFrame()
frames_mean_occ = pd.DataFrame()
frames_std_occ = pd.DataFrame()
frames_pos_occ = pd.DataFrame()
frames_timer = pd.DataFrame()


for f in reversed(os.listdir(base_path + folder)):
    res = ResultFrame(base_path + folder + '/' + f)

    # Without Occlusion
    data = res.delta.mean().values
    data[0] = -data[0]
    frame_mean = pd.DataFrame(
        {f"{f}": data},
        index=res.delta.mean().index)
    frames_mean = frame_mean.join(frames_mean)

    frame_std = pd.DataFrame(
        {f"{f}": res.delta.std().values},
        index=res.delta.mean().index)
    frames_std = frame_std.join(frames_std)
    data = ((res.delta > 0).sum(0) / len(res.delta)).values
    data[0] = 1 - data[0]
    frame_pos = pd.DataFrame(
        {f"{f}": data},
        index=res.delta.mean().index)
    frames_pos = frame_pos.join(frames_pos)

    # With Occlusion
    data = res.delta_occ.mean().values
    data[0] = -data[0]
    frame_mean_occ = pd.DataFrame(
        {f"{f}": data},
        index=res.delta_occ.mean().index)
    frames_mean_occ = frame_mean_occ.join(frames_mean_occ)
    frame_std_occ = pd.DataFrame(
        {f"{f}": res.delta_occ.std().values},
        index=res.delta_occ.mean().index)
    frames_std_occ = frame_std_occ.join(frames_std_occ)
    data = ((res.delta_occ > 0).sum(0) / len(res.delta_occ)).values
    data[0] = 1 - data[0]
    frame_pos_occ = pd.DataFrame(
        {f"{f}": data},
        index=res.delta_occ.mean().index)
    frames_pos_occ = frame_pos_occ.join(frames_pos_occ)

    # Timers
    frame_timer = pd.DataFrame(
        {f"{k}": res.timer.iloc[1][k] for k in res.timer.iloc[1].keys()}, index=[f])
    frames_timer = pd.concat([frames_timer, frame_timer])

if os.path.isfile('methods_comparison.xlsx'):
    folder = 'methods_comparison'
    with pd.ExcelWriter(f'{folder}.xlsx', mode='a') as writer:
        frames_mean.to_excel(writer, sheet_name='Delta moyen LD')
        frames_mean_occ.to_excel(writer, sheet_name='Delta occ moyen LD')
        frames_std.to_excel(writer, sheet_name='Delta std LD')
        frames_std_occ.to_excel(writer, sheet_name='Delta occ std LD')
        frames_pos.to_excel(writer, sheet_name='Delta positif LD')
        frames_pos_occ.to_excel(writer, sheet_name='Delta occ positif LD')
        frames_timer.to_excel(writer, sheet_name='Temps method LD')
else:
    with pd.ExcelWriter(f'{folder}.xlsx') as writer:
        frames_mean.to_excel(writer, sheet_name='Delta moyen LD')
        frames_mean_occ.to_excel(writer, sheet_name='Delta occ moyen LD')
        frames_std.to_excel(writer, sheet_name='Delta std LD')
        frames_std_occ.to_excel(writer, sheet_name='Delta occ std LD')
        frames_pos.to_excel(writer, sheet_name='Delta positif LD')
        frames_pos_occ.to_excel(writer, sheet_name='Delta occ positif LD')
        frames_timer.to_excel(writer, sheet_name='Temps method LD')

time.sleep(1)
