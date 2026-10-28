def answer_spans(text):
    spans = []
    offset = 0
    for line in text.splitlines(keepends=True):
        for start in range(0, len(line), 1400):
            end = min(start + 1600, len(line))
            if line[start:end].strip():
                spans.append(
                    dict(
                        span_id="A" + str(len(spans) + 1),
                        start=offset + start,
                        end=offset + end,
                        text=line[start:end],
                    )
                )
        offset += len(line)
    return spans
