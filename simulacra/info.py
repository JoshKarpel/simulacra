from typing import Any, Tuple, Iterable, Dict, Union


class Info:
    """
    A class that represents a hierarchical collection of information.

    Each :class:`Info` contains a header and a dictionary of children.
    The header is a string that will be written at the top-level of this Info.
    Each child is either a field, which will be written out as ``'{key}: {value}'``, or another Info, which will display itself.

    Field names are unique.
    """

    def __init__(self, *, header: str):
        """
        Parameters
        ----------
        header
            The header for this :class:`Info`.
        """
        self.header = header
        self._children: Dict[Union[str, int], Union[str, Info]] = {}

    def __str__(self) -> str:
        field_strings = [self.header]

        for field, value in self._children.items():
            if isinstance(value, Info):
                info_strings = str(value).split("\n")
                field_strings.append("├─ " + info_strings[0])
                field_strings.extend(
                    "│  " + info_string for info_string in info_strings[1:]
                )
            else:
                field_strings.append(f"├─ {field}: {value}")

        # this loop goes over the field strings in reverse, cleaning up the tail of the structure indicators
        for index, field_string in reversed(list(enumerate(field_strings))):
            if (
                field_string[0] == "├"
            ):  # this is the last branch on this level, replace it with endcap and break
                field_strings[index] = field_string.replace("├", "└")
                break
            else:  # not yet at last branch, continue cleanup
                field_strings[index] = field_string.replace("│", " ", 1)

        return "\n".join(field_strings)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.header})"

    def add_field(self, name: str, value: Any):
        """
        Add a field to the :class:`Info`, which will be displayed as ``'{name}: {value}'``.

        Parameters
        ----------
        name
            The name of the field.
        value
            The value of the field.
        """
        self._children[name] = value

    def rm_field(self, name):
        return self._children.pop(name, None)

    def add_fields(self, name_value_pairs: Iterable[Tuple[str, Any]]):
        """
        Add a list of fields to the :class:`Info`.

        Parameters
        ----------
        name_value_pairs
            An iterable or dict of ``(name, value)`` pairs to add as fields.
        """
        self._children.update(dict(name_value_pairs))

    def add_info(self, info: "Info"):
        """
        Add a sub-Info to the :class:`Info`, which will be displayed at a deeper indentation level.

        Parameters
        ----------
        info
            An :class:`Info` to be added as a sub-Info.
        """
        self._children[id(info)] = info

    def add_infos(self, *infos: "Info"):
        """
        Add a list of Infos to this Info as sub-Infos.

        Parameters
        ----------
        infos
            An iterable of :class:`Info`
        """
        self._children.update({id(info): info for info in infos})
