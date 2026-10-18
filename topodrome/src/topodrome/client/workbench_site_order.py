"""A captured draft for reordering the existing Sites table, without a dialog."""

from dataclasses import dataclass


@dataclass
class SiteOrderDraft:
    generation: int
    original: tuple[str, ...]
    preferences: tuple[str, ...]
    order: list[str]
    sites: dict[str, dict]
    grabbed: str | None = None

    @classmethod
    def capture(cls, policy, sites):
        original = tuple(policy["preferred_site_ids"])
        registered = {row["site_id"]: dict(row) for row in sites}
        order = [site for site in original if site in registered]
        order.extend(site for site in registered if site not in order)
        return cls(policy["generation"], original, original, order, registered)

    @property
    def changed(self):
        return self.preferences != self.original

    def move(self, site_id, direction):
        if site_id not in self.order:
            return False
        index = self.order.index(site_id)
        destination = index + direction
        if not 0 <= destination < len(self.order):
            return False
        self.order[index], self.order[destination] = self.order[destination], self.order[index]
        self.preferences = tuple(self.order)
        return True

    def automatic(self):
        self.preferences = ()
        self.grabbed = None

    def rows(self, current):
        latest = {row["site_id"]: row for row in current}
        rows = [
            latest.get(site_id, {**self.sites[site_id], "state": "unregistered"})
            for site_id in self.order
        ]
        return rows + [row for site_id, row in latest.items() if site_id not in self.sites]
