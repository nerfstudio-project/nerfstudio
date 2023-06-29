# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

import dataclasses
from collections import OrderedDict
from typing import TYPE_CHECKING

import tyro

from nerfstudio.configs.dataparser_configs import dataparser_configs
from nerfstudio.configs.external_methods import get_external_methods
from nerfstudio.configs.method_configs import descriptions as method_descriptions
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.data.dataparsers.base_dataparser import DataParserConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.registry import discover_dataparsers, discover_methods


def merge_methods(methods, method_descriptions, new_methods, new_descriptions, overwrite=True):
    """Merge new methods and descriptions into existing methods and descriptions.
    Args:
        methods: Existing methods.
        method_descriptions: Existing descriptions.
        new_methods: New methods to merge in.
        new_descriptions: New descriptions to merge in.
    Returns:
        Merged methods and descriptions.
    """
    methods = OrderedDict(**methods)
    method_descriptions = OrderedDict(**method_descriptions)
    for k, v in new_methods.items():
        if overwrite or k not in methods:
            methods[k] = v
            method_descriptions[k] = new_descriptions.get(k, "")
    return methods, method_descriptions


def sort_methods(methods, method_descriptions):
    """Sort methods and descriptions by method name."""
    methods = OrderedDict(sorted(methods.items(), key=lambda x: x[0]))
    method_descriptions = OrderedDict(sorted(method_descriptions.items(), key=lambda x: x[0]))
    return methods, method_descriptions


def to_snake_case(name: str) -> str:
    """Convert a name to snake case."""
    return "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")


# DataParsers
external_dataparsers, _external_dataparsers_help = discover_dataparsers()
all_dataparsers = {**dataparser_configs, **external_dataparsers}

# Methods
all_methods, all_descriptions = method_configs, method_descriptions
# Add discovered external methods
discovered_methods, discovered_descriptions = discover_methods()
all_methods, all_descriptions = merge_methods(
    all_methods, all_descriptions, discovered_methods, discovered_descriptions
)
all_methods, all_descriptions = sort_methods(all_methods, all_descriptions)

# Register all possible external methods which can be installed with Nerfstudio
all_methods, all_descriptions = merge_methods(
    all_methods, all_descriptions, *sort_methods(*get_external_methods()), overwrite=False
)

# We also register all external dataparsers found in the external methods
_registered_dataparsers = set(map(type, all_dataparsers.values()))
for method_name, method in discovered_methods.items():
    if (
        hasattr(method.pipeline.datamanager, "dataparser")
        and type(method.pipeline.datamanager.dataparser) not in _registered_dataparsers
    ):
        name = (
            method_name + "-" + to_snake_case(type(method.pipeline.datamanager.dataparser).__name__).replace("_", "-")
        )
        all_dataparsers[name] = method.pipeline.datamanager.dataparser

if TYPE_CHECKING:
    # For static analysis (tab completion, type checking, etc), just use the base
    # dataparser config.
    DataParserUnion = DataParserConfig
else:
    # At runtime, populate a Union type dynamically. This is used by `tyro` to generate
    # subcommands in the CLI.
    DataParserUnion = tyro.extras.subcommand_type_from_defaults(
        all_dataparsers,
        prefix_names=False,  # Omit prefixes in subcommands themselves.
    )

AnnotatedDataParserUnion = tyro.conf.OmitSubcommandPrefixes[DataParserUnion]  # Omit prefixes of flags in subcommands.
"""Union over possible dataparser types, annotated with metadata for tyro. This is
the same as the vanilla union, but results in shorter subcommand names."""


def fix_method_dataparser_type(method_config: TrainerConfig):
    def replace_type(instance, field, value=None, new_type=None):
        assert dataclasses.is_dataclass(instance)
        if value is None:
            value = getattr(instance, field)
        if new_type is None:
            new_type = type(value)
        cls = type(instance)
        fields = dataclasses.fields(cls)
        new_fields = [(field, new_type)]
        config = cls.__dataclass_params__
        new_cls = dataclasses.make_dataclass(
            cls.__name__,
            new_fields,
            bases=(cls,),
            init=config.init,
            repr=config.repr,
            eq=config.eq,
            order=config.order,
            unsafe_hash=config.unsafe_hash,
            frozen=config.frozen,
        )
        kwargs = {i.name: getattr(instance, i.name) for i in fields}
        kwargs[field] = value
        return new_cls(**kwargs)

    if hasattr(method_config.pipeline.datamanager, "dataparser"):
        return replace_type(
            method_config,
            "pipeline",
            replace_type(
                method_config.pipeline,
                "datamanager",
                replace_type(
                    method_config.pipeline.datamanager,
                    "dataparser",
                    new_type=AnnotatedDataParserUnion,
                ),
            ),
        )
    return method_config


for key, method in all_methods.items():
    all_methods[key] = fix_method_dataparser_type(method)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=all_methods, descriptions=all_descriptions)
    ]
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
