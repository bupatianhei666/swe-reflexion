import os

_enc = None

_voyageai = None


def count_tokens(content: str, model: str = 'gpt-3.5-turbo') -> int:
    global _enc, _voyageai

    if model.startswith('voyage'):
        if _voyageai is None:
            voyageai_import_err = (
                '`voyageai` package not found, please run `pip install voyageai`'
            )
            try:
                import voyageai
            except ImportError:
                raise ImportError(voyageai_import_err)

            _voyageai = voyageai.Client()

        return _voyageai.count_tokens([content])

    if _enc is None:
        tiktoken_import_err = (
            '`tiktoken` package not found, please run `pip install tiktoken`'
        )
        try:
            import tiktoken
        except ImportError:
            raise ImportError(tiktoken_import_err)

        # set tokenizer cache temporarily
        should_revert = False
        if 'TIKTOKEN_CACHE_DIR' not in os.environ:
            should_revert = True
            os.environ['TIKTOKEN_CACHE_DIR'] = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '_static/tiktoken_cache',
            )

        # TODO：需要修改tokenizer，使用gpt4o的cl100k_base分词器，原来的tiktoken.encoding_for_model无法正确映射到相应的分词器
        # _enc = tiktoken.encoding_for_model(model)
        _enc = tiktoken.get_encoding("cl100k_base")

        if should_revert:
            del os.environ['TIKTOKEN_CACHE_DIR']

    return len(_enc.encode(content, allowed_special='all'))
