import abc
import re

import bokeh.plotting as bok
from bokeh import layouts as bklayouts
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.io import show
from bokeh.models import HoverTool, Span, RangeSlider, widgets, annotations, TextInput, NormalHead
from bokeh.core.has_props import HasProps
from bokeh.model import Model
from bokeh.core.properties import Bool
from bokeh.models.callbacks import CustomJS
import bokeh.palettes

import itertools

import numpy as np

from scipy import integrate

from .df import find_FWHM


def increment_label(label):
    pattern = r'(\D+)([0-9]*)'

    def repl(match):
        num = match.group(2)
        if num is '':
            num = ' 1'
        else:
            num = str(int(num) + 1)
        return match.group(1) + num
    return re.sub(pattern, repl, label)


class RangeAnalyser:
    analysis_types = []

    def __init__(self, df, ylabel, xlabel=None):
        """
        Class for visualising and performing analysis on range within a set of data.
        Parameters
        ----------
        df : pandas.DataFrame
           DataFrame containing data to analyse
        ylabel: str
            name of column containing y data
        xlabel: str, optional
            name of column containing x data, default is None, in which case uses `df.index`
        """
        self.df = df.copy()
        self.ylabel = ylabel
        self.xlabel = xlabel
        
        self.analyses = {}

        self._start = None
        self._end = None
        self._range_state = RangeState()

        self.gui = RangeAnalyserGui(self)

    @property
    def x(self):
        """
        whole x data
        """
        return self.df[self.xlabel] if self.xlabel else self.df.index

    @property
    def y(self):
        """
        whole y data
        """
        return self.df[self.ylabel]

    @property
    def xrange(self):
        """
        width of x data
        """
        return self.x.values[-1] - self.x.values[0]

    @property
    def yrange(self):
        """
        height of y data
        """
        return self.y.values[-1] - self.y.values[0]

    @property
    def n(self):
        """
        number of datapoints
        """
        return len(self.df)

    @property
    def dx(self):
        """
        average x separation
        """
        return self.xrange / self.n

    @property
    def start(self):
        """
        current start value of slider
        """
        return self._start if self._start else self.gui.start

    @start.setter
    def start(self, val):
        self._start = val

    @property
    def end(self):
        """
        current end value of slider
        """
        return self._end if self._end else self.gui.end

    @end.setter
    def end(self, val):
        self._end = val

    def connect(self):
        self.start = None
        self.end = None

    @property
    def start_connected(self):
        return self._start is None

    @property
    def end_connected(self):
        return self._end is None

    @property
    def range(self):
        """
        subsection of `df` containing data between range of slider
        """
        x = self.x
        return self.df[(x > self.start) & (x < self.end)]

    @property
    def rangex(self):
        """
        x data within slider range
        """
        return self.range[self.xlabel] if self.xlabel else self.range.index

    @property
    def rangey(self):
        """
        y data within slider range
        """
        return self.range[self.ylabel]

    def show(self, notebook_url=None):
        """
        Display the Bokeh app for analysis.

        Parameters
        ----------
        notebook_url: str, optional
            url of notebook, required if not using "http://localhost:8888', passed on to `bokeh.io.show`
        """
        show(self.gui.app, notebook_url)

    @classmethod
    def add_analysis(cls, f):
        """
        Add an analysis to the kit. Can be used as a class decorator to register them.

        Parameters
        ----------
        f: BaseAnalysis
            Subclass of Base analysis to add to RangeAnalyser, if done right should add as an option below the plot
        """
        cls.analysis_types.append(f)

    def __getattr__(self, item):
        if item in self.analyses:
            return self.analyses[item]
        raise AttributeError(item)

    def __repr__(self):
        return (f"<RangeAnalyser " 
                f"x: {self.xlabel if self.xlabel else 'df.index'} y: {self.ylabel}," 
                f" start: {self.start} ({'connected' if self.start_connected else 'disconnected'})"
                f" end: {self.end} ({'connected' if self.end_connected else 'disconnected'})>")


class RangeAnalyserGui:
    move_spans = \
        """
        start_span.location = cb_obj.value[0];
        start_span.change.emit();
        end_span.location = cb_obj.value[1];        
        end_span.change.emit();
        }
        """

    update_range_slide = \
        """
        range_slide.start = cb_obj.start;
        range_slide.end = cb_obj.end;
        range_slide.change.emit();
        """

    def __init__(self, analyser):
        """
        Creates and manages the Gui for RangeAnalysis
        """
        self.analyser = analyser

        self.fig = self.make_fig()

        init_start = self.analyser.x.min() * 1.1
        init_end = self.analyser.x.max() * 0.9

        span_kwargs = dict(dimension='height', line_dash='dashed', line_width=3)
        self.start_span = Span(location=init_start, line_color='green', **span_kwargs)
        self.end_span = Span(location=init_end, line_color='red', **span_kwargs)

        self.range_slide = RangeSlider(start=self.analyser.x.min(), end=self.analyser.x.max(), step=self.analyser.dx,
                                       value=(init_start, init_end),
                                       width=self.fig.plot_width)

        self.start_connected = widgets.Button(label="start", button_type='success')
        self.end_connected = widgets.Button(label="end", button_type='success')

        self.setup_callbacks()

        self.app = self.make_app()
        self.doc = None

    @property
    def start(self):
        """
        current start value of slider
        """
        return self.range_slide.value[0]

    @start.setter
    def start(self, val):
        old = self.range_slide.value
        new = (val, old[1])
        self.range_slide.value = new
        #self.range_slide.trigger('value', old, new)
        old_loc = self.start_span.location
        self.start_span.location= val
        # self.start_span.trigger('location', old_loc, val)

    @property
    def end(self):
        """
        current end value of slider
        """
        return self.range_slide.value[1]

    @end.setter
    def end(self, val):
        old = self.range_slide.value
        new = (old[0], val)
        self.range_slide.value = new
        #self.range_slide.trigger('value', old, new)
        old_loc = self.end_span.location

        self.end_span.location = val
        # self.end_span.trigger('location', old_loc, val)

    def make_fig(self):
        fig = bok.figure()

        fig.tools.append(HoverTool(
            tooltips=[(f"{self.analyser.xlabel if self.analyser.xlabel else 'x'}, {self.analyser.ylabel}",
                       "$x, $y")]))
        fig.line(self.analyser.x, self.analyser.y, legend=self.analyser.ylabel)

        return fig

    def setup_callbacks(self):
        self.range_slide.js_on_change('value', CustomJS(code=self.move_spans,
                                                        args={'start_span': self.start_span,
                                                              'end_span': self.end_span}))
        self.fig.x_range.callback = CustomJS(code=self.update_range_slide,
                                        args={'range_slide': self.range_slide})
        self.start_connected.on_click(self.toggle_start_connected)
        self.end_connected.on_click(self.toggle_end_connected)
                                        

    def make_app(self):
        self.fig.add_layout(self.start_span)
        self.fig.add_layout(self.end_span)

        analysis_guis = []

        for Analysis in self.analyser.analysis_types:
            analysis = Analysis(self.analyser)
            self.analyser.analyses[analysis.name] = analysis
            analysis_guis.append(analysis.gui.as_row())

        layout = bklayouts.layout(
            [self.fig, bklayouts.column(*analysis_guis, bklayouts.row(self.start_connected,
                                                                      self.end_connected, width=300))],
                                   self.range_slide)

        def modify_doc(doc):
            self.doc = doc
            doc.add_root(layout)
            doc.add_periodic_callback(self.update_connected, 100)

        handler = FunctionHandler(modify_doc)
        app = Application(handler)
        return app

    def update_connected(self):
        old_start, old_end = self.start_connected.button_type, self.end_connected.button_type
        self.start_connected.button_type = "success" if self.analyser.start_connected else "danger"
        self.end_connected.button_type = "success" if self.analyser.end_connected else "danger"
    
    def toggle_start_connected(self):
        if self.analyser.start_connected:
            self.analyser.start = self.start
        else:
            self.analyser.start = None
        self.update_connected()
        
    def toggle_end_connected(self):
        if self.analyser.end_connected:
            self.analyser.end = self.end
        else:
            self.analyser.end = None
        self.update_connected()
    
            
        

class BaseAnalysisGui:
    """
    Baseclass for analysis
    """

    def __init__(self, analyser, analysis):
        self.analyser = analyser
        self.analysis = analysis

        self.label_input = TextInput(value=f'{analysis.short} 1')
        self.do_button = widgets.Button(label=f'Run {analysis.name}')
        self.do_button.on_click(self.run)

    def run(self):
        result = self.analysis.run()
        self.analysis.results[self.label_input.value] = result
        self.annotate()
        self.label_input.value = self.next_label

    @property
    def current_label(self):
        return self.label_input.value

    @property
    def next_label(self):
        return increment_label(self.current_label)

    def annotate(self):
        """
        Subclass to implement drawing of your analysis, will be called after clicking run button.

        Notes
        -----
        For some reason `fig.add_layout` doesn't seem to work, instead things like annotations should directly be added
        to `fig.renderers`.
        """
        pass

    def as_row(self, width=300):
        return bklayouts.row(self.label_input, self.do_button, width=width)


class BaseAnalysis:
    """
    Baseclass for analysis to be performed on the range selected by RangeAnalyser
    """
    name = 'analysis_name'
    gui_cls = BaseAnalysisGui

    def __init__(self, analyser):
        self.analyser = analyser
        self.results = {}
        self.gui = self.gui_cls(analyser, self)

    @abc.abstractmethod
    def run(self):
        pass

    def mrun(self, label, start, end):
        """
        Manually run the analysis basically emulates clicking on Run with the range set to start, end
        """
        old_start, old_end = self.analyser._start, self.analyser._end
        self.analyser.start = start
        self.analyser.end = end

        result = self.run()

        # otherwise would move slider unexpectedly
        self.analyser.start, self.analyser.end = old_start, old_end

        self.results[label] = result
        return result

    def __getitem__(self, item):
        return self.results[item]


class MaximaGui(BaseAnalysisGui):

    def annotate(self):
        max_x_y = self.analysis.results[self.current_label]
        max_x, max_y = max_x_y

        x_start = max_x - self.analyser.xrange * 0.1

        arrow = annotations.Arrow(x_start=x_start, y_start=max_y,
                                  x_end=max_x, y_end=max_y)
        label = annotations.Label(text=f"{self.current_label}: {max_y:.1f}", x=x_start, y=max_y, text_align='right')

        self.analyser.gui.fig.renderers.extend([arrow, label])


@RangeAnalyser.add_analysis
class Maxima(BaseAnalysis):
    """
    Find the maximum value within the range
    """
    name = 'maxima'
    short = 'max'
    gui_cls = MaximaGui

    def run(self):
        """
        Find the maximum value within the range
        Returns
        -------
        max_xy, max_y : tuple(float, float)
            Coordinates of maximmum y position within range
        """
        max_x_y = (self.analyser.rangey.idxmax(), self.analyser.rangey.max())
        return max_x_y


class FWHMGui(BaseAnalysisGui):

    def annotate(self):
        FWHM, (x_left, x_right), HM = self.analysis.results[self.current_label]

        arrow = annotations.Arrow(x_start=x_left, y_start=HM,
                                  x_end=x_right, y_end=HM, start=NormalHead(), end=NormalHead())
        label = annotations.Label(text=f"{self.current_label}: {FWHM:.1f}", x=(x_left + x_right) / 2, y=HM, text_align='center')
        self.analyser.gui.fig.renderers.extend([arrow, label])


@RangeAnalyser.add_analysis
class FWHM(BaseAnalysis):
    """
    Find the FWHM (full width at half maximum) of peaks within the range
    """
    name = 'FWHM'
    short = 'FWHM'
    gui_cls = FWHMGui

    def run(self):
        """
        Find the FWHM (full width at half maximum) of peaks within the range

        Returns
        -------
        FWHM : float
            Width of peak at half maximum AKA FWHM
        (x_left, x_right) : tuple
            xlabel coords of boundary of peak
        HM : float
            Half maximum value
        """
        return find_FWHM(self.analyser.range, self.analyser.ylabel, self.analyser.xlabel)


class IntegrateGui(BaseAnalysisGui):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pallet = itertools.cycle(bokeh.palettes.Category10[10])

    def annotate(self):
        A, (start, end) = self.analysis.results[self.current_label]
        x = self.analyser.rangex
        y = self.analyser.rangey
        self.analyser.gui.fig.varea(x=x, y1=np.zeros_like(y), y2=y, color=next(self.pallet))
        label = annotations.Label(text=f"{self.current_label}: {A:.1f}", x=(start + end) / 2, y=0, text_align='center')
        self.analyser.gui.fig.renderers.append(label)


@RangeAnalyser.add_analysis
class Integrate(BaseAnalysis):
    name = 'integrate'
    short ='int'

    gui_cls = IntegrateGui

    def run(self):
        """
        Use `scipy.integrate.trapz`i.e. the trapezoid method!

        Returns
        -------
        A : float
            Integral over the range
        (start, end): tuple(float, float)
            x coordinates integrated over
        """
        an = self.analyser
        return integrate.trapz(an.rangey, an.rangex), (an.start, an.end)
