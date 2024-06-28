# -*- coding: utf-8 -*-
import argparse
import os
import random
import re

import yaml


def get_cur_path():
    """Get the absolute path of current file.

    Returns: The absolute path of this file (Config.py).

    """
    return os.path.dirname(__file__)


DEFAULT_FILE = os.path.join(get_cur_path(), "default.yaml")


class Config(object):  # 继承 object 类，定义一个新式类
    """
    `LibFewShot` 的配置解析器。

    `Config` 用于解析 *.yaml 文件、控制台参数和 run_*.py 设置，并将其合并为 Python 字典。合并冲突的解决规则如下：

    1. 合并是递归的，如果一个键没有被指定，将使用现有的值。
    2. 合并优先级为：控制台参数 > run_*.py 字典 > 用户定义的 yaml (/LibFewShot/config/*.yaml) > default.yaml (/LibFewShot/core/config/default.yaml)
    """

    def __init__(self, config_file=None, variable_dict=None, is_resume=False):
        """
        初始化参数字典，实际上完成了所有参数定义的合并。

        参数:
            config_file: 配置文件名。(/LibFewShot/config/name.yaml)
            variable_dict: 变量字典。
            is_resume: 指定是否恢复，默认为 False。
        """
        self.is_resume = is_resume  # 是否恢复训练
        self.config_file = config_file  # 配置文件名
        self.console_dict = self._load_console_dict()  # 加载控制台参数
        self.default_dict = self._load_config_files(DEFAULT_FILE)  # 加载默认配置文件
        self.file_dict = self._load_config_files(config_file)  # 加载指定的配置文件
        self.variable_dict = self._load_variable_dict(variable_dict)  # 加载变量字典
        self.config_dict = self._merge_config_dict()  # 合并配置字典

    def get_config_dict(self):
        """
        返回合并后的配置字典。

        返回:
            dict: 一个包含 LibFewShot 设置的字典。
        """
        return self.config_dict

    @staticmethod
    def _load_config_files(config_file):
        """
        解析 YAML 文件。

        参数:
            config_file (str): yaml 文件路径。

        返回:
            dict: 一个包含 LibFewShot 设置的字典。
        """
        config_dict = dict()
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
                     [-+]?[0-9][0-9_]*\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                    |[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+
                    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                    |[-+]?\\.(?:inf|Inf|INF)
                    |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )

        if config_file is not None:
            with open(config_file, "r", encoding="utf-8") as fin:
                config_dict.update(yaml.load(fin.read(), Loader=loader))
        config_file_dict = config_dict.copy()
        for include in config_dict.get("includes", []):
            # 此处会将"./config/"拼接到路径中
            with open(os.path.join("./config/", include), "r", encoding="utf-8") as fin:
                config_dict.update(yaml.load(fin.read(), Loader=loader))
        if config_dict.get("includes") is not None:
            config_dict.pop("includes")
        config_dict.update(config_file_dict)
        return config_dict

    @staticmethod
    def _load_variable_dict(variable_dict):
        """
        从 run_*.py 加载变量字典。

        参数:
            variable_dict (dict): 配置字典。

        返回:
            dict: 一个包含 LibFewShot 设置的字典。
        """
        config_dict = dict()
        config_dict.update(variable_dict if variable_dict is not None else {})
        return config_dict

    @staticmethod
    def _load_console_dict():
        """
        解析命令行参数

        返回:
            dict: 一个包含 LibFewShot 控制台设置的字典。
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("-w", "--way_num", type=int, help="way num")
        parser.add_argument("-s", "--shot_num", type=int, help="shot num")
        parser.add_argument("-q", "--query_num", type=int, help="query num")
        parser.add_argument("-bs", "--batch_size", type=int, help="batch_size")
        parser.add_argument("-es", "--episode_size", type=int, help="episode_size")

        parser.add_argument("-data", "--data_root", help="dataset path")
        parser.add_argument(
            "-log_name",
            "--log_name",
            help="specific log dir name if necessary",
        )
        parser.add_argument("-image_size", type=int, help="image size")
        parser.add_argument("-aug", "--augment", type=bool, help="use augment or not")
        parser.add_argument(
            "-aug_times",
            "--augment_times",
            type=int,
            help="augment times (for support in few-shot)",
        )
        parser.add_argument(
            "-aug_times_query",
            "--augment_times_query",
            type=int,
            help="augment times for query in few-shot",
        )
        parser.add_argument("-train_episode", type=int, help="train episode num")
        parser.add_argument("-test_episode", type=int, help="test episode num")
        parser.add_argument("-epochs", type=int, help="epoch num")
        parser.add_argument("-result", "--result_root", help="result path")
        parser.add_argument("-save_interval", type=int, help="checkpoint save interval")
        parser.add_argument(
            "-log_level",
            help="log level in: debug, info, warning, error, critical",
        )
        parser.add_argument("-log_interval", type=int, help="log interval")
        parser.add_argument("-gpus", "--device_ids", help="device ids")
        # TODO: n_gpu should be len(gpus)?
        parser.add_argument("-n_gpu", type=int, help="gpu num")
        parser.add_argument("-seed", type=int, help="seed")
        parser.add_argument("-deterministic", type=bool, help="deterministic or not")
        parser.add_argument("-tag", "--tag", type=str, help="experiment tag")
        args = parser.parse_args()
        # 删除键值为 None 的键值对
        return {k: v for k, v in vars(args).items() if v is not None}

    def _recur_update(self, dic1, dic2):
        """
        递归合并字典。

        用于递归地合并两个字典（配置文件），`dic2` 将覆盖 `dic1` 中相同键的值。

        参数:
            dic1 (dict): 将被覆盖的字典。（低优先级）
            dic2 (dict): 覆盖的字典。（高优先级）

        返回:
            dict: 合并后的字典。
        """
        if dic1 is None:
            dic1 = dict()
        for k in dic2.keys():
            if isinstance(dic2[k], dict):
                dic1[k] = self._recur_update(
                    dic1[k] if k in dic1.keys() else None, dic2[k]
                )
            else:
                dic1[k] = dic2[k]
        return dic1

    def _update(self, dic1, dic2):
        """
        合并字典。

        用于合并两个字典（配置文件），`dic2` 将覆盖 `dic1` 中相同键的值。

        参数:
            dic1 (dict): 将被覆盖的字典。（低优先级）
            dic2 (dict): 覆盖的字典。（高优先级）

        返回:
            dict: 合并后的字典。
        """
        if dic1 is None:
            dic1 = dict()
        for k in dic2.keys():
            dic1[k] = dic2[k]
        return dic1

    def _merge_config_dict(self):
        """
        合并所有字典。

        1. 合并是递归的，如果一个键没有被指定，将使用现有的值。
        2. 合并优先级为：控制台参数 > run_*.py 字典 > 用户定义的 yaml (/LibFewShot/config/*.yaml) > default.yaml (/LibFewShot/core/config/default.yaml)

        返回:
            dict: 一个包含 LibFewShot 设置的字典。
        """
        config_dict = dict()
        config_dict = self._update(config_dict, self.default_dict)
        config_dict = self._update(config_dict, self.file_dict)
        config_dict = self._update(config_dict, self.variable_dict)
        config_dict = self._update(config_dict, self.console_dict)

        # 如果 test_* 未定义，用 *_num 代替
        if config_dict["test_way"] is None:
            config_dict["test_way"] = config_dict["way_num"]
        if config_dict["test_shot"] is None:
            config_dict["test_shot"] = config_dict["shot_num"]
        if config_dict["test_query"] is None:
            config_dict["test_query"] = config_dict["query_num"]
        if config_dict["port"] is None:
            port = random.randint(25000, 55000)
            while self.is_port_in_use("127.0.0.1", port):
                old_port = port
                port = str(int(port) + 1)
                print(
                    "Warning: Port {} is already in use, switch to port {}".format(
                        old_port, port
                    )
                )
            config_dict["port"] = port

        # 修改或添加一些配置
        config_dict["resume"] = self.is_resume
        if self.is_resume:
            config_dict["resume_path"] = self.config_file[: -1 * len("/config.yaml")]
        config_dict["tb_scale"] = (
            float(config_dict["train_episode"]) / config_dict["test_episode"]
        )

        return config_dict

    def is_port_in_use(self, host, port):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, int(port))) == 0
