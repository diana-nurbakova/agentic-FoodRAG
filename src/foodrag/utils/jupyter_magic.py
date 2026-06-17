# Define the custom magic command
import sys
import traceback
from IPython.core import magic_arguments
from IPython.core.magic import register_cell_magic
from IPython.display import display, Markdown
from io import StringIO
from tqdm.notebook import tqdm

INTERNAL_ERROR_PARAM_SUBSTRING = 'error_ioawhfn89123'
NO_CAPTURED_OUTPUT_MESSAGE = '~~No captured output~~'
LOG_FILE_NAME = 'logfile'
DEFAULT_CAPTURE_VAR_NAME = 'capt'


class capture_stdout:
    """
    Context manager similar to `contextlib.redirect_stdout`, but
    different in that it:
      - temporarily writes stdout to other streams *in addition to*
        rather than *instead of* `sys.stdout`
      - accepts any number of streams and sends stdout to each
      - can optionally keep streams open after exiting context by
        passing `closing=False`

    Parameters
    ----------
    *streams : *`io.IOBase`
        stream(s) to receive data sent to `sys.stdout`

    closing : `bool`, optional
        if [default: `True`], close streams upon exiting the context
        block.
    """

    def __init__(self, *streams, closing=True):
        self.streams = streams
        self.closing = closing
        self.sys_stdout_write = sys.stdout.write

    def __enter__(self):
        sys.stdout.write = self._write
        if len(self.streams) == 1:
            return self.streams[0]
        return self.streams

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.write = self.sys_stdout_write
        if self.closing:
            for s in self.streams:
                s.close()

    def _write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception as e:
                # If data not encodable, we print error instead of data
                s.write(str(e))

            if hasattr(s, 'flush') and hasattr(s, 'seek') and hasattr(s, 'tell'):
                s.flush()
        self.sys_stdout_write(data)
        sys.stdout.flush()


@magic_arguments.magic_arguments()
@magic_arguments.argument('output', type=str, default='', nargs='?',
                          help="""The name of the variable in which to store output.
    This is a utils.io.CapturedIO object with stdout/err attributes
    for the text of the captured output.
    CapturedOutput also has a show() method for displaying the output,
    and __call__ as well, so you can use that to quickly display the
    output.
    If unspecified, captured output is called as given in DEFAULT_CAPTURE_VAR_NAME.
    """
                          )
@register_cell_magic
def capture_output_and_exceptions(line, cell):
    args = magic_arguments.parse_argstring(capture_output_and_exceptions, line)

    if not args.output:
        args.output = DEFAULT_CAPTURE_VAR_NAME

    # Execute the cell and always capture to the regular sys.stdout. In addition, stores the output two additional
    # streams: A String Stream to store the output in a variable and in a file stream which is stored.
    with capture_stdout(StringIO(), open(LOG_FILE_NAME + '_' + args.output + '.txt', 'w')) as (io_stream, file_stream):
        execution_result = get_ipython().run_cell(cell)
        output = io_stream.getvalue()

        if execution_result is not None:
            # Store the captured output in the variable named by the user
            get_ipython().user_ns[args.output] = output

            if not execution_result.success:
                try:
                    execution_result.raise_error()
                except Exception as e:
                    # Get the traceback
                    full_traceback = traceback.format_exception(type(e), e, e.__traceback__)

                    # Remove the magic paths (elements with indices 1, 2, and 3) from the traceback
                    tb_lines = [full_traceback[0]] + full_traceback[4:]

                    # Store the traceback in error variable
                    get_ipython().user_ns[args.output + INTERNAL_ERROR_PARAM_SUBSTRING] = '\n'.join(tb_lines)

                    # Add the traceback to the captured file output
                    file_stream.write('\n' + '\n'.join(tb_lines))


def out(out_param, include_last_error=True):
    # Print the captured output if there is any
    try:
        captured_stdout = get_ipython().user_ns[out_param]
    except KeyError:
        captured_stdout = None

    if captured_stdout is not None:
        print(captured_stdout, file=sys.stdout)
    else:
        print(NO_CAPTURED_OUTPUT_MESSAGE, file=sys.stdout)

    # Print the captured error if there is one
    if include_last_error:
        try:
            error = get_ipython().user_ns[out_param + INTERNAL_ERROR_PARAM_SUBSTRING]
        except KeyError:
            error = None

        if error:
            print(error, file=sys.stderr)


def as_markdown(text, bold=True):
    if bold:
        display(Markdown(f"<b>{text}</b>"))
    else:
        display(Markdown(f"{text}"))


class DummyTqdm:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def __iter__(self):
        return iter(range(10))

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def disable_tqdm():
    tqdm.tqdm = DummyTqdm


def enable_tqdm():
    tqdm.tqdm = tqdm  # Restore the original tqdm class
