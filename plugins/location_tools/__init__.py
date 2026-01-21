from dataclasses import dataclass

from plugins.location_tools import locationtools
from plugins.requirement import PluginRequirement #, Plugin, 

# dataclass自动为类生成特殊方法，包括构造函数、__repr__以及__eq__方法
@dataclass
class LocationToolsRequirement(PluginRequirement):
    name: str = 'location_tools'
    documentation: str = locationtools.DOCUMENTATION


# class LocationToolsPlugin(Plugin):
#     name: str = 'location_tools'
