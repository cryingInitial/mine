import logging

from methods.er_new import ER
from methods.baseline_new import BASELINE
from methods.sdp_new import SDP
from methods.der_new import DER
from methods.ewc_new import EWCpp
from methods.ours_new import Ours
from methods.mir_new import MIR
from methods.aser_new import ASER
from methods.bic_new import BiasCorrection
from methods.etf import ETF
from methods.etf_er import ETF_ER
from methods.etf_er_ce import ETF_ER_CE
from methods.twf_new import TWF
from methods.baseline_new_joint import BASELINE_JOINT
logger = logging.getLogger()


def select_method(args, train_datalist, test_datalist, device):
    kwargs = vars(args)
    print("!!", args.mode == "er")
    if args.mode == "er":
        method = ER(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "baseline_joint":
        method = BASELINE_JOINT(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "etf_er":
        method = ETF_ER(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "etf_er_ce":
        method = ETF_ER_CE(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "bic":
        method = BiasCorrection(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "mir":
        method = MIR(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "baseline":
        method = BASELINE(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "etf":
        method = ETF(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "aser":
        method = ASER(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "ewc":
        method = EWCpp(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    # elif args.mode == "gdumb":
    #     from methods.gdumb import GDumb
    #     method = GDumb(
    #         train_datalist=train_datalist,
    #         test_datalist=test_datalist,
    #         cls_dict=cls_dict,
    #         device=device,
    #         **kwargs,
    #     )
    # elif args.mode == "mir":
    #     method = MIR(
    #         train_datalist=train_datalist,
    #         test_datalist=test_datalist,
    #         cls_dict=cls_dict,
    #         device=device,
    #         **kwargs,
    #     )
    # elif args.mode == "clib":
    #     method = CLIB(
    #         train_datalist=train_datalist,
    #         test_datalist=test_datalist,
    #         cls_dict=cls_dict,
    #         device=device,
    #         **kwargs,
    #     )
    elif args.mode == "der":
        method = DER(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "sdp":
        method = SDP(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "twf":        
        method = TWF(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
        
    elif args.mode == "ours":
        method = Ours(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    else:
        print("??", args.mode)
        raise NotImplementedError("Choose the args.mode in [er, gdumb, rm, bic, ewc++, mir, clib]")

    return method
