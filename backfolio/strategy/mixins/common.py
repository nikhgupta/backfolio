class StateMixin(object):
    """
    Mixin that allows a strategy to store states into session files.
    """
    def __init__(self, *args, **kwargs):
        self._state = []
        self._state_fields = []
        super().__init__(*args, **kwargs)
        self.session_fields += ['state']
        for field in self._state_fields:
            setattr(self, "_%s" % field, None)

    def reset(self, context):
        """
        Resets the component for reuse.
        Load current state from session at each reset.
        """
        super().reset(context)
        self._load_state()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, val):
        self._state = val

    @property
    def current_state(self):
        return self.state[-1] if len(self.state) > 0 else {}

    def get_state_for(self, field):
        return getattr(self, "_%s" % field)

    def set_state_for(self, field, val, save=None):
        setattr(self, "_%s" % field, val)
        if save is not None:
            self._save_state(**save)
        return val

    def _load_state(self):
        """
        Load current state of session.
        """
        for field in self._state_fields:
            if field in self.current_state:
                self.set_state_for(field, self.current_state[field])
        return self.current_state

    def _save_state(self, **data):
        """
        Save current state of session and any given data at this time.
        Specifically, add `_last_rebalance` time to session state.
        """
        extra = {field: self.get_state_for(field) for field in self._state_fields}
        data = {**data, **extra, **{"time": self.tick.time}}
        self._state.append(data)



class SelectedSymbolsMixin(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._selected_symbols = []

    @property
    def symbols(self):
        return self._selected_symbols

    @symbols.setter
    def symbols(self, arr=[]):
        self._selected_symbols = arr

    def transform_history(self, panel):
        """
        Transform history such that it only contains specified symbols
        """
        if self.symbols:
            panel = panel[self.symbols].dropna(how='all')
        panel = super().transform_history(panel)
        return panel


class DataCleanerMixin(object):
    def transform_tick_data(self, data):
        """
        By default, drop any asset without any data, and
        ignore tick data for assets without any volume.
        """
        data = super().transform_tick_data(data)
        data = data.dropna(how='all')
        data = data[data['volume'] > 0]
        return data
