
ALL_STRATEGIES = ['constant', 'linear', 'quadratic', 'exponential']


class FrozenStrategy:

    name: str

    N0: int
    base: float | None = None
    inc: float | None = None
    quad: float | None = None

    def __init__(self,
                 name: str,
                 N0: int = 1,
                 base: float | None = None,
                 inc: float | None = None,
                 quad: float | None = None):
        assert name in ALL_STRATEGIES

        self.name = name
        self.N0 = N0

        if name == 'constant':
            self.Nk_func = lambda k: N0

        elif name == 'exponential':
            assert base is not None
            self.Nk_func = lambda k: int(N0 * base ** k)

        elif name == 'linear':
            assert inc is not None
            self.Nk_func = lambda k: int(N0 + inc * k)

        elif name == 'quadratic':
            assert quad is not None
            self.Nk_func = lambda k: int(N0 + quad * k ** 2)

        else:
            raise ValueError(f"Unknown strategy name: {name}")

        self.base = base
        self.inc = inc
        self.quad = quad

    def forward(self, k: int) -> int:
        return self.Nk_func(k)

    def __call__(self, k: int) -> int:
        return self.Nk_func(k)
