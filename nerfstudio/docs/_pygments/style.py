"""Custom Pygments styles for the Sphinx documentation."""

from pygments.style import Style
from pygments.token import (
    Comment,
    Error,
    Generic,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Token,
    Whitespace,
)


class NerfstudioStyleLight(Style):
    """
    A style based on the manni pygments style.
    """

    background_color = "#f8f9fb"

    styles = {
        Whitespace: "#bbbbbb",
        Comment: "italic #d34600",
        Comment.Preproc: "noitalic #009999",
        Comment.Special: "bold",
        Keyword: "bold #006699",
        Keyword.Pseudo: "nobold",
        Keyword.Type: "#007788",
        Operator: "#555555",
        Operator.Word: "bold #000000",
        Name.Builtin: "#336666",
        Name.Function: "#CC00FF",
        Name.Class: "bold #00AA88",
        Name.Namespace: "bold #00CCFF",
        Name.Exception: "bold #CC0000",
        Name.Variable: "#003333",
        Name.Constant: "#336600",
        Name.Label: "#9999FF",
        Name.Entity: "bold #999999",
        Name.Attribute: "#330099",
        Name.Tag: "bold #330099",
        Name.Decorator: "#9999FF",
        String: "#CC3300",
        String.Doc: "italic",
        String.Interpol: "#AA0000",
        String.Escape: "bold #CC3300",
        String.Regex: "#33AAAA",
        String.Symbol: "#FFCC33",
        String.Other: "#CC3300",
        Number: "#FF6600",
        Generic.Heading: "bold #003300",
        Generic.Subheading: "bold #003300",
        Generic.Deleted: "border:#CC0000 bg:#FFCCCC",
        Generic.Inserted: "border:#00CC00 bg:#CCFFCC",
        Generic.Error: "#FF0000",
        Generic.Emph: "italic",
        Generic.Strong: "bold",
        Generic.Prompt: "bold #000099",
        Generic.Output: "#AAAAAA",
        Generic.Traceback: "#99CC66",
        Error: "bg:#FFAAAA #AA0000",
    }


class NerfstudioStyleDark(Style):
    """
    A style based on the one-dark style.
    """

    background_color = "#282C34"

    styles = {
        Token: "#ABB2BF",
        Punctuation: "#ABB2BF",
        Punctuation.Marker: "#ABB2BF",
        Keyword: "#C678DD",
        Keyword.Constant: "#fdd06c",
        Keyword.Declaration: "#C678DD",
        Keyword.Namespace: "#C678DD",
        Keyword.Reserved: "#C678DD",
        Keyword.Type: "#fdd06c",
        Name: "#ff8c58",
        Name.Attribute: "#ff8c58",
        Name.Builtin: "#fdd06c",
        Name.Class: "#fdd06c",
        Name.Function: "bold #61AFEF",
        Name.Function.Magic: "bold #56B6C2",
        Name.Other: "#ff8c58",
        Name.Tag: "#ff8c58",
        Name.Decorator: "#61AFEF",
        Name.Variable.Class: "",
        String: "#bde3a1",
        Number: "#D19A66",
        Operator: "#56B6C2",
        Comment: "#7F848E",
    }
