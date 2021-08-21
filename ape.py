# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
_DESCRIPTION = """\
fuck you asshole!
"""
import datasets
label2int = {'O': 0, 'I-Reply': 2, 'B-Reply': 1, 'B-Review': 1, 'I-Review': 2}
_TRAINING_FILE = "train.txt"
_DEV_FILE = "dev.txt"
_TEST_FILE = "test.txt"
logger = datasets.logging.get_logger(__name__)


class APEConfig(datasets.BuilderConfig):
    """BuilderConfig for Conll2003"""

    def __init__(self, **kwargs):
        """BuilderConfig forConll2003.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(APEConfig, self).__init__(**kwargs)


class APE(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""
    BUILDER_CONFIGS = [
        APEConfig(name="APE", version=datasets.Version("1.11.0"), description="APE dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Sequence(datasets.Value("string")),
                    "label": datasets.Sequence(datasets.Value("uint32")),
                    "label_pair": datasets.Sequence(datasets.Value("uint32")),
                    "l_review": datasets.Value("uint32"),
                    "l_reply": datasets.Value("uint32"),
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_files = {
            "train": f"{_TRAINING_FILE}",
            "dev": f"{_DEV_FILE}",
            "test": f"{_TEST_FILE}",
        }
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("Generating examples from = %s", filepath)
        with open(filepath, 'r', encoding="utf-8") as f:
            guid = 0
            text = []
            label = []
            label_pair = []
            l_review = 0
            l_reply = 0
            for line in f.readlines():
                line = line.strip()
                if line:  # 以空行作为不同paragraph的分割
                    line = line.split('\t')
                    text.append(line[0])
                    label.append(label2int[line[1]])
                    label_pair.append(int(line[2][2:]) if line[2] != 'O' else 0)
                    if line[-2] == 'Review':
                        l_review += 1
                    else:
                        l_reply += 1
                else:  # 遇到空行，清空所有列表，所以这些是列表的列表
                    if text:
                        yield guid, {
                            "id": str(guid),
                            "text": text,
                            "label": label,
                            "label_pair": label_pair,
                            "l_review": l_review,
                            "l_reply": l_reply,
                        }
                        guid += 1
                        text = []
                        label = []
                        label_pair = []
                        l_review = 0
                        l_reply = 0
# yield guid, {
#     "id": str(guid),
#     "text": text,
#     "label": label,
#     "label_pair": label_pair,
#     "l_review": l_review,
#     "l_reply": l_reply,
# }
# max_len_review = 174  # 174  trial - 11
# max_len_reply = 85  # 85  trial - 10