from typing import Dict, List

from habitat_sim.physics import ContactPointData


def get_contact_infos(
    contact_points: List[ContactPointData],
    obj_id: int,
    link_ids=None,
    exclude_self_collision=True,
) -> List[Dict]:
    info = []
    for x in contact_points:
        if x.object_id_a == obj_id:
            if link_ids is not None and x.link_id_a not in link_ids:
                continue
            if exclude_self_collision and x.object_id_b == obj_id:
                continue
            info.append(
                {
                    "object_id": x.object_id_b,
                    "object_link_id": x.link_id_b,
                    "normal_force": x.normal_force,
                    "link_id": x.link_id_a,
                }
            )
        elif x.object_id_b == obj_id:
            if link_ids is not None and x.link_id_b not in link_ids:
                continue
            if exclude_self_collision and x.object_id_a == obj_id:
                continue
            info.append(
                {
                    "object_id": x.object_id_a,
                    "object_link_id": x.link_id_a,
                    "normal_force": x.normal_force,
                    "link_id": x.link_id_b,
                }
            )
        else:
            pass
    return info


def contact2str(c: ContactPointData):
    keys = [x for x in dir(c) if not x.startswith("__")]
    return ",".join(["{}={}".format(k, getattr(c, k)) for k in keys])
