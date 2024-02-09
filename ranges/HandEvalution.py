class HandEvalution:
    def get_hit_percent(self):
        return self.hit[:, 0].sum() * 100 / self.hit.shape[0]

    def __str__(self):
        return f"{self.name}: {self.get_equity()}"
