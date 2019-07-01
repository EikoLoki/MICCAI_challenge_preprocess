"""
Created on 15 November 2015
By Joe Bell
"""
import sys
#returns python version (3, 2 or 1)
python_version = sys.version_info.major
if python_version == 3:
    from tkinter import *
elif python_version == 2:
    from Tkinter import *
else:
    ImportError("At least Python 2 required.")
import string

class GuiError(Exception):
    """Display a error related to the gui."""
    pass

class Window(Frame):
    """Represent a window(Frame), which you can add widgets to."""
    def __init__(self, title, layout="pack", master=None):
        Frame.__init__(self, master)
        if layout=="pack":
            self.method = "pack"
            self.pack()
        elif layout=="grid":
            self.method = "grid"
            self.grid()
        self.master.title(title)
        self.widgets = []

    def add_button(self, label, layout={}, **kwargs):
        """Add a button to the window"""
        btn  = Button(self, text=label, **kwargs)
        self._place(btn, layout)
        self.widgets.append(btn)
        return btn

    def add_textbox(self, layout={}, **kwargs):
        """Add a textbox to the window"""
        box = Entry(self, **kwargs)
        self._place(box, layout)
        self.widgets.append(box)
        return box

    def add_widget(self, widget, layout={}):
        """Add a widget to the window."""
        self._place(widget, layout)
        self.widgets.append(widget)
        return widget

    def _place(self, widget, layout):
        """Secret"""
        if self.method == "pack":
            widget.pack(**layout)
        elif self.method == "grid":
            widget.grid(**layout)

    def start(self):
        """Start the mainloop."""
        self.mainloop()
        self.started = True
    def stop(self):
        """Close the application."""
        self.master.destroy()

def on(event, widget, function):
    """Bind an event on a widget."""
    events = {
        "click":"<Button-1>",
        "dblClick":"<Double-Button-1>",
        "rightClick":"<Button-3>",
        "hover":"<Enter>",
        "hoveroff":"<Leave>",
        "keyPressEnter":"<Return>"
        }
    for key in list(string.ascii_lowercase+string.digits):
        events["keyPress"+key] = key
    if event == "btnPress" and isinstance(widget, Button):
        widget["command"] = function
    elif event in events:
        widget.bind(events[event], function)
    else:
        try:
            widget.bind(event, function)
        except TclError:
            raise GuiError("The Event is not valid. Maybe try using a click event?")
