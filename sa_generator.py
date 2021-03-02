# coding=utf-8
# Copyright 2021 jemix.
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

# Lint as: python
""" Task: Sentiment Analysis Dataset """

import datasets
import logging

logger = logging.getLogger(__name__)


_CITATION = """\
@inproceedings{id_citation,
    title = "",
    author = "",
    booktitle = "",
    year = "",
    url = "",
    pages = "00--00",
}
"""

_DESCRIPTION = """\
description about data 

For more details see https://github.com/jemiaymen/TC/sa/
"""


_URL = "data/"
_TRAINING_FILE = "train.txt"
_DEV_FILE = "valid.txt"


class SAConfig(datasets.BuilderConfig):
    """BuilderConfig for Sentiment Analysis"""

    def __init__(self, **kwargs):
        """BuilderConfig Sentiment Analysis

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SAConfig, self).__init__(**kwargs)


class SentimentAnalysis(datasets.GeneratorBasedBuilder):
    """Sentiment Analysis dataset."""

    BUILDER_CONFIGS = [
        SAConfig(name="sentiment_analysis", version=datasets.Version(
            "1.0.0"), description="S.A dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(
                        names=[
                            'NEG',  # Negatif Sentiment
                            'POS',  # Positif Sentiment
                        ]
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/jemiaymen/TC/sa/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            "dev": f"{_URL}{_DEV_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                                    "filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={
                                    "filepath": downloaded_files["dev"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            for i, line in enumerate(f):
                splits = line.split("\t")
                if len(splits) < 2:
                    continue

                yield i, {
                    "text": splits[0],
                    "label": splits[1].rstrip(),
                }
        logger.info("Generating examples finish !")
