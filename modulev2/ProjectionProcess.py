from module.BaseModule import BaseModule


class ProjectionProcess(BaseModule):

    def __init__(self, config):
        super(ProjectionProcess, self).__init__(config)

    def _update_conf(self, config, *args, **kwargs):
        self.process = self.define_target_ref(config)

    @staticmethod
    def define_target_ref(config):
        case_dict = {}
        if not config["use_pos"]:
            case_dict['step1'] = 98
            case_dict['step2'] = 99
        else:
            case_dict['step1'] = 1 if config['proj_right'] else 0
            if config["type"] == '2vis':
                if config["position_setup"][1] < 0:
                    case_dict['step2'] = 10
                elif config["position_setup"][1] < config["position_setup"][0]:
                    case_dict['step2'] = 14 if config["proj_right"] else 13
                elif config["position_setup"][1] > config["position_setup"][0]:
                    case_dict['step2'] = 17
                else:
                    case_dict['step2'] = 99
            if config["type"] == '2ir':
                if config["position_setup"][1] < 0:
                    case_dict['step2'] = 11
                elif config["position_setup"][1] < config["position_setup"][0]:
                    case_dict['step2'] = 15 if config["proj_right"] else 12
                elif config["position_setup"][1] > config["position_setup"][0]:
                    case_dict['step2'] = 16
                else:
                    case_dict['step2'] = 99
        return case_dict
