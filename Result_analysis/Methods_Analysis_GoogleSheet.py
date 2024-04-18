import os
import gspread
import pandas as pd
from gspread import Spreadsheet
from gspread_dataframe import set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials
from tqdm import tqdm
from Result_analysis.ResultFrame import ResultFrame


def select_sheet(sh: Spreadsheet, name: str, cols=3, rows=50):
    sheet_names = [s.title for s in spreadsheet.worksheets()]
    if name in sheet_names:
        return sh.worksheet(name)
    else:
        return sh.add_worksheet(name, cols=cols, rows=rows, index=len(sh.worksheets()))


if __name__ == '__main__':
    scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("pipelinetestresult-e441b2c7d865.json", scope)
    client = gspread.authorize(creds)

    spreadsheet = client.open("Result ProcessPipe methods")

    ## Result for daytime #################
    folder = "methods_comparison"
    base_path = os.getcwd() + "/../results/"
    frames_mean = pd.DataFrame()
    frames_std = pd.DataFrame()
    frames_pos = pd.DataFrame()
    frames_mean_occ = pd.DataFrame()
    frames_std_occ = pd.DataFrame()
    frames_pos_occ = pd.DataFrame()
    frames_timer = pd.DataFrame()
    for f in tqdm(reversed(os.listdir(base_path + folder)), desc='DataFrames creation'):
        res = ResultFrame(base_path + folder + '/' + f)

        # Without Occlusion
        data = res.delta.mean().values
        data[0] = -data[0]  # To inverse the sign for RMSE delta
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
        data[0] = -data[0]  # To inverse the sign for RMSE delta
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

    result = {'Timer methods': frames_timer,
              'Delta moyen': frames_mean,
              'Delta occ moyen': frames_mean_occ,
              'Delta std': frames_std,
              'Delta occ std': frames_std_occ,
              'Delta positif': frames_pos,
              'Delta occ positif': frames_pos_occ}
    for name, val in tqdm(result.items(), desc='Recording into GoogleSheet'):
        worksheet = select_sheet(spreadsheet, name)
        set_with_dataframe(worksheet, val, include_index=True)

    ## Result for nighttime #################
    folder = "methods_comparison_night"
    base_path = os.getcwd() + "/../results/"
    frames_mean = pd.DataFrame()
    frames_std = pd.DataFrame()
    frames_pos = pd.DataFrame()
    frames_mean_occ = pd.DataFrame()
    frames_std_occ = pd.DataFrame()
    frames_pos_occ = pd.DataFrame()
    frames_timer = pd.DataFrame()
    for f in tqdm(reversed(os.listdir(base_path + folder)), desc='DataFrames creation'):
        res = ResultFrame(base_path + folder + '/' + f)

        # Without Occlusion
        data = res.delta.mean().values
        data[0] = -data[0]  # To inverse the sign for RMSE delta
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
        data[0] = -data[0]  # To inverse the sign for RMSE delta
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

    result = {'Timer methods Night': frames_timer,
              'Delta moyen Night': frames_mean,
              'Delta occ moyen Night': frames_mean_occ,
              'Delta std Night': frames_std,
              'Delta occ std Night': frames_std_occ,
              'Delta positif Night': frames_pos,
              'Delta occ positif Night': frames_pos_occ}
    for name, val in tqdm(result.items(), desc='Recording into GoogleSheet'):
        worksheet = select_sheet(spreadsheet, name)
        set_with_dataframe(worksheet, val, include_index=True)
