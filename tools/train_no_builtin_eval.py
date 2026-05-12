import train


def _skip_builtin_eval(*args, **kwargs):
    print("Skip built-in eval; run validation/AP from the wrapper script instead.")


train.repeat_eval_ckpt = _skip_builtin_eval


if __name__ == "__main__":
    train.main()
