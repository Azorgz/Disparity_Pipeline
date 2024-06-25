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
    frame_choice = 'cumroi'
    frames_mean = pd.DataFrame()
    frames_std = pd.DataFrame()
    frames_pos = pd.DataFrame()
    frames_timer = pd.DataFrame()

    for f in tqdm(reversed(os.listdir(base_path + folder)), desc='DataFrames creation'):
        res = ResultFrame(base_path + folder + '/' + f)

        if frame_choice == 'occ':
            data_frame = res.delta_occ
        elif frame_choice == 'roi':
            data_frame = res.delta_roi
        elif frame_choice == 'cumroi':
            data_frame = res.delta_cumroi
        else:
            data_frame = res.delta
        # Delta mean ----------------------------------------------------------------
        data = data_frame.mean().values
        data[0] = -data[0]  # To inverse the sign for RMSE delta
        frames_mean = pd.DataFrame(
            {f"{f}": data},
            index=data_frame.mean().index).join(frames_mean)

        # Delta std ----------------------------------------------------------------
        frames_std = pd.DataFrame(
            {f"{f}": data_frame.std().values},
            index=data_frame.mean().index).join(frames_std)

        # Delta pos ----------------------------------------------------------------
        data = ((data_frame > 0).sum(0) / len(data_frame)).values
        data[0] = 1 - data[0]
        frames_pos = pd.DataFrame(
            {f"{f}": data},
            index=data_frame.mean().index).join(frames_pos)

        # Computing time ----------------------------------------------------------------
        frames_timer = pd.concat([frames_timer, pd.DataFrame(
            {f"{k}": res.timer.iloc[1][k] for k in res.timer.iloc[1].keys()}, index=[f])])

    result = {'Timer methods': frames_timer,
              'Delta moyen': frames_mean,
              'Delta std': frames_std,
              'Delta positif': frames_pos}
    for name, val in tqdm(result.items(), desc='Recording into GoogleSheet'):
        worksheet = select_sheet(spreadsheet, name)
        set_with_dataframe(worksheet, val, include_index=True)
    #
    # ## Result for nighttime #################
    # folder = "methods_comparison_night"
    # base_path = os.getcwd() + "/../results/"
    # frames_mean = pd.DataFrame()
    # frames_std = pd.DataFrame()
    # frames_pos = pd.DataFrame()
    # frames_timer = pd.DataFrame()
    # for f in tqdm(reversed(os.listdir(base_path + folder)), desc='DataFrames creation'):
    #     res = ResultFrame(base_path + folder + '/' + f)
    #
    #     if frame_choice == 'occ':
    #         data_frame = res.delta_occ
    #     elif frame_choice == 'roi':
    #         data_frame = res.delta_roi
    #     elif frame_choice == 'cumroi':
    #         data_frame = res.delta_cumroi
    #     else:
    #         data_frame = res.delta
    #     # Delta mean ----------------------------------------------------------------
    #     data = data_frame.mean().values
    #     data[0] = -data[0]  # To inverse the sign for RMSE delta
    #     frames_mean = pd.DataFrame(
    #         {f"{f}": data},
    #         index=data_frame.mean().index).join(frames_mean)
    #
    #     # Delta std ----------------------------------------------------------------
    #     frames_std = pd.DataFrame(
    #         {f"{f}": data_frame.std().values},
    #         index=data_frame.mean().index).join(frames_std)
    #
    #     # Delta pos ----------------------------------------------------------------
    #     data = ((data_frame > 0).sum(0) / len(data_frame)).values
    #     data[0] = 1 - data[0]
    #     frames_pos = pd.DataFrame(
    #         {f"{f}": data},
    #         index=data_frame.mean().index).join(frames_pos)
    #
    #     # Computing time ----------------------------------------------------------------
    #     frames_timer = pd.concat([frames_timer, pd.DataFrame(
    #         {f"{k}": res.timer.iloc[1][k] for k in res.timer.iloc[1].keys()}, index=[f])])
    #
    # result = {'Timer methods Night': frames_timer,
    #           'Delta moyen Night': frames_mean,
    #           'Delta std Night': frames_std,
    #           'Delta positif Night': frames_pos}
    # for name, val in tqdm(result.items(), desc='Recording into GoogleSheet'):
    #     worksheet = select_sheet(spreadsheet, name)
    #     set_with_dataframe(worksheet, val, include_index=True)
