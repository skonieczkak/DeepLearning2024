from contextlib import redirect_stdout
import io

def capture_model_summary(model):
    with io.StringIO() as buf, redirect_stdout(buf):
        model.summary()
        return buf.getvalue()

def print_model_summaries_side_by_side(models):
    summaries = [capture_model_summary(model).splitlines() for model in models]
    max_length = max(len(summary) for summary in summaries)
    
    # Ensure all summaries have the same number of lines
    for summary in summaries:
        summary += [''] * (max_length - len(summary))
    
    # Print summaries side by side
    for lines in zip(*summaries):
        print(" | ".join(f"{line:<80}" for line in lines))