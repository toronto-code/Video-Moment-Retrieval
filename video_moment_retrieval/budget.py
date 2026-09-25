"""Reserve HTTP attempts for later ingestion stages without increasing the user cap."""
from __future__ import annotations


class StageBudgets:
    def __init__(self, stages: list[tuple[str, object, int]]):
        self.order = [name for name, _, _ in stages]
        self.clients = {}
        for name, provider, weight in stages:
            client = getattr(provider, 'client', None)
            if client is None or not hasattr(client, 'stage_limit'):
                continue
            group = self.clients.setdefault(id(client), {'client': client, 'previous': client.stage_limit, 'weights': {}})
            group['weights'][name] = weight
        self.allocations = {}
        for group in self.clients.values():
            client = group['client']
            maximum = min(client.max_requests, group['previous'] if group['previous'] is not None else client.max_requests)
            available = max(0, maximum-client.attempts)
            weights = group['weights']
            quota = {name: 0 for name in weights}
            # Even a tiny remaining budget reserves one attempt for publishing vectors.
            if 'embedding' in quota and available:
                quota['embedding'] = 1
                available -= 1
            denominator = sum(weights.values())
            shares = {name: available*weight/denominator for name, weight in weights.items()}
            for name, share in shares.items():
                quota[name] += int(share)
            remainder = available-sum(int(share) for share in shares.values())
            for name in sorted(weights, key=lambda name: -(shares[name]-int(shares[name])))[:remainder]:
                quota[name] += 1
            group.update(maximum=maximum, quota=quota)
            for name, value in quota.items():
                self.allocations[name] = self.allocations.get(name, 0)+value

    def enter(self, stage: str) -> None:
        later = set(self.order[self.order.index(stage)+1:])
        for group in self.clients.values():
            reserved = sum(value for name, value in group['quota'].items() if name in later)
            group['client'].stage_limit = group['maximum']-reserved

    def restore(self) -> None:
        for group in self.clients.values():
            group['client'].stage_limit = group['previous']
