import os
import time
import pandas as pd
from Result_analysis.ResultFrame import ResultFrame


folder = "methods_comparison"
base_path = os.getcwd() + "/../results/"

frames_mean = pd.DataFrame()
frames_std = pd.DataFrame()
frames_pos = pd.DataFrame()

frames_mean_occ = pd.DataFrame()
frames_std_occ = pd.DataFrame()
frames_pos_occ = pd.DataFrame()

frames_mean_roi = pd.DataFrame()
frames_std_roi = pd.DataFrame()
frames_pos_roi = pd.DataFrame()

frames_mean_cumroi = pd.DataFrame()
frames_std_cumroi = pd.DataFrame()
frames_pos_cumroi = pd.DataFrame()

frames_timer = pd.DataFrame()

res_available = []
for f in reversed(os.listdir(base_path + folder)):
    res = ResultFrame(base_path + folder + '/' + f)
    res_available.extend(res.available_res)
    # Without Occlusion ----------------------------------------------------------------
    data = res.delta.mean().values
    data[0] = -data[0]  # To inverse the sign for RMSE delta
    frames_mean = pd.DataFrame(
        {f"{f}": data},
        index=res.delta.mean().index).join(frames_mean)
    frames_std = pd.DataFrame(
        {f"{f}": res.delta.std().values},
        index=res.delta.mean().index).join(frames_std)
    data = ((res.delta > 0).sum(0) / len(res.delta)).values
    data[0] = 1 - data[0]
    frames_pos = pd.DataFrame(
        {f"{f}": data},
        index=res.delta.mean().index).join(frames_pos)

    # With Occlusion ----------------------------------------------------------------
    if 'delta_occ' in res.available_res:
        data = res.delta_occ.mean().values
        data[0] = -data[0]  # To inverse the sign for RMSE delta
        frames_mean_occ = pd.DataFrame(
            {f"{f}": data},
            index=res.delta_occ.mean().index).join(frames_mean_occ)
        frames_std_occ = pd.DataFrame(
            {f"{f}": res.delta_occ.std().values},
            index=res.delta_occ.mean().index).join(frames_std_occ)
        data = ((res.delta_occ > 0).sum(0) / len(res.delta_occ)).values
        data[0] = 1 - data[0]
        frames_pos_occ = pd.DataFrame(
            {f"{f}": data},
            index=res.delta_occ.mean().index).join(frames_pos_occ)

    # With ROI ----------------------------------------------------------------
    if 'delta_roi' in res.available_res:
        data = res.delta_roi.mean().values
        data[0] = -data[0]  # To inverse the sign for RMSE delta
        frames_mean_roi = pd.DataFrame(
            {f"{f}": data},
            index=res.delta_roi.mean().index).join(frames_mean_roi)
        frames_std_roi = pd.DataFrame(
            {f"{f}": res.delta_roi.std().values},
            index=res.delta_roi.mean().index).join(frames_std_roi)
        data = ((res.delta_roi > 0).sum(0) / len(res.delta_roi)).values
        data[0] = 1 - data[0]
        frames_pos_roi = pd.DataFrame(
            {f"{f}": data},
            index=res.delta_roi.mean().index).join(frames_pos_roi)

    # With CUMULATIVE ROI ----------------------------------------------------------------
    if 'delta_cumroi' in res.available_res:
        data = res.delta_cumroi.mean().values
        data[0] = -data[0]  # To inverse the sign for RMSE delta
        frames_mean_cumroi = pd.DataFrame(
            {f"{f}": data},
            index=res.delta_cumroi.mean().index).join(frames_mean_cumroi)
        frames_std_cumroi = pd.DataFrame(
            {f"{f}": res.delta_cumroi.std().values},
            index=res.delta_cumroi.mean().index).join(frames_std_cumroi)
        data = ((res.delta_cumroi > 0).sum(0) / len(res.delta_cumroi)).values
        data[0] = 1 - data[0]
        frames_pos_cumroi = pd.DataFrame(
            {f"{f}": data},
            index=res.delta_cumroi.mean().index).join(frames_pos_cumroi)

    # Timers
    frame_timer = pd.DataFrame(
        {f"{k}": res.timer.iloc[1][k] for k in res.timer.iloc[1].keys()}, index=[f])
    frames_timer = pd.concat([frames_timer, frame_timer])


with pd.ExcelWriter(f'{folder}.xlsx') as writer:
    frames_mean.to_excel(writer, sheet_name='Delta moyen')
    frames_std.to_excel(writer, sheet_name='Delta std')
    frames_pos.to_excel(writer, sheet_name='Delta positif')

    if 'delta_occ' in res_available:
        frames_mean_occ.to_excel(writer, sheet_name='Delta occ moyen')
        frames_std_occ.to_excel(writer, sheet_name='Delta occ std')
        frames_pos_occ.to_excel(writer, sheet_name='Delta occ positif')

    if 'delta_occ' in res_available:
        frames_mean_occ.to_excel(writer, sheet_name='Delta occ moyen')
        frames_std_occ.to_excel(writer, sheet_name='Delta occ std')
        frames_pos_occ.to_excel(writer, sheet_name='Delta occ positif')

    frames_timer.to_excel(writer, sheet_name='Temps method')

time.sleep(1)
