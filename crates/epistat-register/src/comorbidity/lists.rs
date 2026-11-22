//! The code lists and weights of the comorbidity indices, written by
//! `tests/data/comorbidity_lists.R` from the maps of the R package
//! comorbidity 1.1.0 (R R version 4.6.2 (2026-10-29)). Do not edit by hand.

/// A condition as the package holds it: key, label and ICD-10 prefixes.
pub(super) struct Spec {
    pub key: &'static str,
    pub label: &'static str,
    pub prefixes: &'static [&'static str],
}

pub(super) static CHARLSON_QUAN_ICD10: [Spec; 17] = [
    Spec {
        key: "mi",
        label: "Myocardial infarction",
        prefixes: &["I21", "I22", "I252"],
    },
    Spec {
        key: "chf",
        label: "Congestive heart failure",
        prefixes: &[
            "I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426", "I427", "I428", "I429",
            "I43", "I50", "P290",
        ],
    },
    Spec {
        key: "pvd",
        label: "Peripheral vascular disease",
        prefixes: &[
            "I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792", "K551", "K558", "K559",
            "Z958", "Z959",
        ],
    },
    Spec {
        key: "cevd",
        label: "Cerebrovascular disease",
        prefixes: &[
            "G45", "G46", "H340", "I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68",
            "I69",
        ],
    },
    Spec {
        key: "dementia",
        label: "Dementia",
        prefixes: &["F00", "F01", "F02", "F03", "F051", "G30", "G311"],
    },
    Spec {
        key: "cpd",
        label: "Chronic pulmonary disease",
        prefixes: &[
            "I278", "I279", "J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47", "J60", "J61",
            "J62", "J63", "J64", "J65", "J66", "J67", "J684", "J701", "J703",
        ],
    },
    Spec {
        key: "rheumd",
        label: "Rheumatoid disease",
        prefixes: &[
            "M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360",
        ],
    },
    Spec {
        key: "pud",
        label: "Peptic ulcer disease",
        prefixes: &["K25", "K26", "K27", "K28"],
    },
    Spec {
        key: "mld",
        label: "Mild liver disease",
        prefixes: &[
            "B18", "K700", "K701", "K702", "K703", "K709", "K713", "K714", "K715", "K717", "K73",
            "K74", "K760", "K762", "K763", "K764", "K768", "K769", "Z944",
        ],
    },
    Spec {
        key: "diab",
        label: "Diabetes without chronic complications",
        prefixes: &[
            "E100", "E101", "E106", "E108", "E109", "E110", "E111", "E116", "E118", "E119", "E120",
            "E121", "E126", "E128", "E129", "E130", "E131", "E136", "E138", "E139", "E140", "E141",
            "E146", "E148", "E149",
        ],
    },
    Spec {
        key: "diabwc",
        label: "Diabetes with chronic complications",
        prefixes: &[
            "E102", "E103", "E104", "E105", "E107", "E112", "E113", "E114", "E115", "E117", "E122",
            "E123", "E124", "E125", "E127", "E132", "E133", "E134", "E135", "E137", "E142", "E143",
            "E144", "E145", "E147",
        ],
    },
    Spec {
        key: "hp",
        label: "Hemiplegia or paraplegia",
        prefixes: &[
            "G041", "G114", "G801", "G802", "G81", "G82", "G830", "G831", "G832", "G833", "G834",
            "G839",
        ],
    },
    Spec {
        key: "rend",
        label: "Renal disease",
        prefixes: &[
            "I120", "I131", "N032", "N033", "N034", "N035", "N036", "N037", "N052", "N053", "N054",
            "N055", "N056", "N057", "N18", "N19", "N250", "Z490", "Z491", "Z492", "Z940", "Z992",
        ],
    },
    Spec {
        key: "canc",
        label: "Cancer (any malignancy)",
        prefixes: &[
            "C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11",
            "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23",
            "C24", "C25", "C26", "C30", "C31", "C32", "C33", "C34", "C37", "C38", "C39", "C40",
            "C41", "C43", "C45", "C46", "C47", "C48", "C49", "C50", "C51", "C52", "C53", "C54",
            "C55", "C56", "C57", "C58", "C60", "C61", "C62", "C63", "C64", "C65", "C66", "C67",
            "C68", "C69", "C70", "C71", "C72", "C73", "C74", "C75", "C76", "C81", "C82", "C83",
            "C84", "C85", "C88", "C90", "C91", "C92", "C93", "C94", "C95", "C96", "C97",
        ],
    },
    Spec {
        key: "msld",
        label: "Moderate or severe liver disease",
        prefixes: &[
            "I850", "I859", "I864", "I982", "K704", "K711", "K721", "K729", "K765", "K766", "K767",
        ],
    },
    Spec {
        key: "metacanc",
        label: "Metastatic solid tumour",
        prefixes: &["C77", "C78", "C79", "C80"],
    },
    Spec {
        key: "aids",
        label: "AIDS/HIV",
        prefixes: &["B20", "B21", "B22", "B24"],
    },
];

pub(super) static ELIXHAUSER_QUAN_ICD10: [Spec; 31] = [
    Spec {
        key: "chf",
        label: "Congestive heart failure",
        prefixes: &[
            "I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426", "I427", "I428", "I429",
            "I43", "I50", "P290",
        ],
    },
    Spec {
        key: "carit",
        label: "Cardiac arrhythmias",
        prefixes: &[
            "I441", "I442", "I443", "I456", "I459", "I47", "I48", "I49", "R000", "R001", "R008",
            "T821", "Z450", "Z950",
        ],
    },
    Spec {
        key: "valv",
        label: "Valvular disease",
        prefixes: &[
            "A520", "I05", "I06", "I07", "I08", "I091", "I098", "I34", "I35", "I36", "I37", "I38",
            "I39", "Q230", "Q231", "Q232", "Q233", "Z952", "Z953", "Z954",
        ],
    },
    Spec {
        key: "pcd",
        label: "Pulmonary circulation disorders",
        prefixes: &["I26", "I27", "I280", "I288", "I289"],
    },
    Spec {
        key: "pvd",
        label: "Peripheral vascular disorders",
        prefixes: &[
            "I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792", "K551", "K558", "K559",
            "Z958", "Z959",
        ],
    },
    Spec {
        key: "hypunc",
        label: "Hypertension, uncomplicated",
        prefixes: &["I10"],
    },
    Spec {
        key: "hypc",
        label: "Hypertension, complicated",
        prefixes: &["I11", "I12", "I13", "I15"],
    },
    Spec {
        key: "para",
        label: "Paralysis",
        prefixes: &[
            "G041", "G114", "G801", "G802", "G81", "G82", "G830", "G831", "G832", "G833", "G834",
            "G839",
        ],
    },
    Spec {
        key: "ond",
        label: "Other neurological disorders",
        prefixes: &[
            "G10", "G11", "G12", "G13", "G20", "G21", "G22", "G254", "G255", "G312", "G318",
            "G319", "G32", "G35", "G36", "G37", "G40", "G41", "G931", "G934", "R470", "R56",
        ],
    },
    Spec {
        key: "cpd",
        label: "Chronic pulmonary disease",
        prefixes: &[
            "I278", "I279", "J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47", "J60", "J61",
            "J62", "J63", "J64", "J65", "J66", "J67", "J684", "J701", "J703",
        ],
    },
    Spec {
        key: "diabunc",
        label: "Diabetes, uncomplicated",
        prefixes: &[
            "E100", "E101", "E109", "E110", "E111", "E119", "E120", "E121", "E129", "E130", "E131",
            "E139", "E140", "E141", "E149",
        ],
    },
    Spec {
        key: "diabc",
        label: "Diabetes, complicated",
        prefixes: &[
            "E102", "E103", "E104", "E105", "E106", "E107", "E108", "E112", "E113", "E114", "E115",
            "E116", "E117", "E118", "E122", "E123", "E124", "E125", "E126", "E127", "E128", "E132",
            "E133", "E134", "E135", "E136", "E137", "E138", "E142", "E143", "E144", "E145", "E146",
            "E147", "E148",
        ],
    },
    Spec {
        key: "hypothy",
        label: "Hypothyroidism",
        prefixes: &["E00", "E01", "E02", "E03", "E890"],
    },
    Spec {
        key: "rf",
        label: "Renal failure",
        prefixes: &[
            "I120", "I131", "N18", "N19", "N250", "Z490", "Z491", "Z492", "Z940", "Z992",
        ],
    },
    Spec {
        key: "ld",
        label: "Liver disease",
        prefixes: &[
            "B18", "I85", "I864", "I982", "K70", "K711", "K713", "K714", "K715", "K717", "K72",
            "K73", "K74", "K760", "K762", "K763", "K764", "K765", "K766", "K767", "K768", "K769",
            "Z944",
        ],
    },
    Spec {
        key: "pud",
        label: "Peptic ulcer disease excluding bleeding",
        prefixes: &[
            "K257", "K259", "K267", "K269", "K277", "K279", "K287", "K289",
        ],
    },
    Spec {
        key: "aids",
        label: "AIDS/HIV",
        prefixes: &["B20", "B21", "B22", "B24"],
    },
    Spec {
        key: "lymph",
        label: "Lymphoma",
        prefixes: &[
            "C81", "C82", "C83", "C84", "C85", "C88", "C96", "C900", "C902",
        ],
    },
    Spec {
        key: "metacanc",
        label: "Metastatic cancer",
        prefixes: &["C77", "C78", "C79", "C80"],
    },
    Spec {
        key: "solidtum",
        label: "Solid tumour without metastasis",
        prefixes: &[
            "C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11",
            "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23",
            "C24", "C25", "C26", "C30", "C31", "C32", "C33", "C34", "C37", "C38", "C39", "C40",
            "C41", "C43", "C45", "C46", "C47", "C48", "C49", "C50", "C51", "C52", "C53", "C54",
            "C55", "C56", "C57", "C58", "C60", "C61", "C62", "C63", "C64", "C65", "C66", "C67",
            "C68", "C69", "C70", "C71", "C72", "C73", "C74", "C75", "C76", "C97",
        ],
    },
    Spec {
        key: "rheumd",
        label: "Rheumatoid artritis/collaged vascular disease",
        prefixes: &[
            "L940", "L941", "L943", "M05", "M06", "M08", "M120", "M123", "M30", "M310", "M311",
            "M312", "M313", "M32", "M33", "M34", "M35", "M45", "M461", "M468", "M469",
        ],
    },
    Spec {
        key: "coag",
        label: "Coagulopathy",
        prefixes: &[
            "D65", "D66", "D67", "D68", "D691", "D693", "D694", "D695", "D696",
        ],
    },
    Spec {
        key: "obes",
        label: "Obesity",
        prefixes: &["E66"],
    },
    Spec {
        key: "wloss",
        label: "Weight loss",
        prefixes: &[
            "E40", "E41", "E42", "E43", "E44", "E45", "E46", "R634", "R64",
        ],
    },
    Spec {
        key: "fed",
        label: "Fluid and electrolyte disorders",
        prefixes: &["E222", "E86", "E87"],
    },
    Spec {
        key: "blane",
        label: "Blood loss anaemia",
        prefixes: &["D500"],
    },
    Spec {
        key: "dane",
        label: "Deficiency anaemia",
        prefixes: &["D508", "D509", "D51", "D52", "D53"],
    },
    Spec {
        key: "alcohol",
        label: "Alcohol abuse",
        prefixes: &[
            "F10", "E52", "G621", "I426", "K292", "K700", "K703", "K709", "T51", "Z502", "Z714",
            "Z721",
        ],
    },
    Spec {
        key: "drug",
        label: "Drug abuse",
        prefixes: &[
            "F11", "F12", "F13", "F14", "F15", "F16", "F18", "F19", "Z715", "Z722",
        ],
    },
    Spec {
        key: "psycho",
        label: "Psychoses",
        prefixes: &[
            "F20", "F22", "F23", "F24", "F25", "F28", "F29", "F302", "F312", "F315",
        ],
    },
    Spec {
        key: "depre",
        label: "Depression",
        prefixes: &[
            "F204", "F313", "F314", "F315", "F32", "F33", "F341", "F412", "F432",
        ],
    },
];

pub(super) const CHARLSON_1987: [f64; 17] = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 6.0, 6.0,
];

pub(super) const CHARLSON_QUAN_2011: [f64; 17] = [
    0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 4.0, 6.0, 4.0,
];

pub(super) const ELIXHAUSER_VAN_WALRAVEN_2009: [f64; 31] = [
    7.0, 5.0, -1.0, 4.0, 2.0, 0.0, 0.0, 7.0, 6.0, 3.0, 0.0, 0.0, 0.0, 5.0, 11.0, 0.0, 0.0, 9.0,
    12.0, 4.0, 0.0, 3.0, -4.0, 6.0, 5.0, -2.0, -2.0, 0.0, -7.0, 0.0, -3.0,
];
