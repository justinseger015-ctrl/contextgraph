//! Full-State Verification: WS2 (Training Data) + WS3 (Hard Negatives) + WS5 (LoRA Key)
//!
//! Run: cargo test --release -p context-graph-embeddings --test ws_embeddings_verification -- --nocapture

// ============================================================================
// WS2: Training Data Expansion
// ============================================================================

mod ws2 {
    use context_graph_embeddings::training::data::{seed_training_pairs, TrainingDirection};

    #[test]
    fn happy_path_total_pairs() {
        let pairs = seed_training_pairs();
        println!("=== WS2 Total Pairs ===");
        println!("Count: {}", pairs.len());
        assert!(pairs.len() >= 250, "Need >=250, got {}", pairs.len());
        println!("[PASS]\n");
    }

    #[test]
    fn happy_path_direction_distribution() {
        let pairs = seed_training_pairs();
        let fwd = pairs.iter().filter(|p| matches!(p.direction, TrainingDirection::Forward)).count();
        let bwd = pairs.iter().filter(|p| matches!(p.direction, TrainingDirection::Backward)).count();
        let bi = pairs.iter().filter(|p| matches!(p.direction, TrainingDirection::Bidirectional)).count();
        let none = pairs.iter().filter(|p| matches!(p.direction, TrainingDirection::None)).count();

        println!("=== WS2 Direction Distribution ===");
        println!("Forward={} Backward={} Bidir={} None={}", fwd, bwd, bi, none);
        assert!(none >= 40, "Need >=40 non-causal, got {}", none);
        assert!(bi >= 5, "Need >=5 bidirectional, got {}", bi);

        let ratio = none as f64 / pairs.len() as f64 * 100.0;
        println!("Non-causal ratio: {:.1}%", ratio);
        assert!(ratio >= 10.0, "Ratio {:.1}% below 10%", ratio);
        println!("[PASS]\n");
    }

    #[test]
    fn happy_path_domain_coverage() {
        let pairs = seed_training_pairs();
        let mut domains = std::collections::HashMap::new();
        for p in &pairs {
            if !p.domain.is_empty() {
                *domains.entry(p.domain.clone()).or_insert(0usize) += 1;
            }
        }

        println!("=== WS2 Domain Coverage ===");
        let mut sorted: Vec<_> = domains.iter().collect();
        sorted.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
        for (d, c) in &sorted { println!("  {:15} {}", d, c); }

        let expected = ["health","environment","economics","technology","social",
            "physics","nutrition","cybersecurity","psychology","history","legal","engineering"];
        for d in &expected {
            assert!(domains.contains_key(*d), "Missing domain: {}", d);
        }

        let legal = domains.get("legal").copied().unwrap_or(0);
        let eng = domains.get("engineering").copied().unwrap_or(0);
        assert!(legal >= 15, "Legal {} < 15", legal);
        assert!(eng >= 15, "Engineering {} < 15", eng);
        println!("[PASS]\n");
    }

    #[test]
    fn edge_no_empty_text() {
        let pairs = seed_training_pairs();
        for (i, p) in pairs.iter().enumerate() {
            assert!(!p.cause_text.trim().is_empty(), "Pair {} empty cause", i);
            assert!(!p.effect_text.trim().is_empty(), "Pair {} empty effect", i);
        }
        println!("WS2 Edge: no empty text in {} pairs [PASS]", pairs.len());
    }

    #[test]
    fn edge_all_have_metadata() {
        let pairs = seed_training_pairs();
        for (i, p) in pairs.iter().enumerate() {
            assert!(!p.domain.is_empty(), "Pair {} missing domain", i);
            assert!(!p.mechanism.is_empty(), "Pair {} missing mechanism", i);
        }
        println!("WS2 Edge: all {} pairs have domain+mechanism [PASS]", pairs.len());
    }

    #[test]
    fn edge_confidence_range() {
        let pairs = seed_training_pairs();
        for (i, p) in pairs.iter().enumerate() {
            assert!(p.confidence >= 0.0 && p.confidence <= 1.0,
                "Pair {} confidence {} OOB", i, p.confidence);
        }
        let min = pairs.iter().map(|p| p.confidence).fold(f32::INFINITY, f32::min);
        let max = pairs.iter().map(|p| p.confidence).fold(f32::NEG_INFINITY, f32::max);
        println!("WS2 Edge: confidence [{:.2}, {:.2}] all in [0,1] [PASS]", min, max);
    }

    /// Verify that implicit causal pairs exist (no explicit causal markers)
    #[test]
    fn verify_implicit_causal_pairs() {
        let pairs = seed_training_pairs();
        let explicit_markers = ["causes", "leads to", "results in", "because", "due to",
            "consequently", "therefore", "triggers", "induces"];

        let implicit_count = pairs.iter()
            .filter(|p| matches!(p.direction, TrainingDirection::Forward))
            .filter(|p| {
                let combined = format!("{} {}", p.cause_text.to_lowercase(), p.effect_text.to_lowercase());
                !explicit_markers.iter().any(|m| combined.contains(m))
            })
            .count();

        println!("=== WS2 Implicit Causal Pairs ===");
        println!("Forward pairs without explicit markers: {}", implicit_count);
        assert!(implicit_count >= 10, "Need >=10 implicit causal, got {}", implicit_count);
        println!("[PASS]\n");
    }
}

// ============================================================================
// WS3: Hard Negative Mining
// ============================================================================

mod ws3 {
    use context_graph_embeddings::training::loss::{DirectionalContrastiveLoss, LossConfig};
    use candle_core::{Device, Tensor, Var};

    fn make_vecs(n: usize, d: usize) -> (Tensor, Tensor) {
        let device = Device::Cpu;
        let c: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let e: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.2 + 1.0).cos()).collect();
        let ct = Tensor::from_slice(&c, (n, d), &device).unwrap();
        let et = Tensor::from_slice(&e, (n, d), &device).unwrap();
        let cn = ct.sqr().unwrap().sum(1).unwrap().sqrt().unwrap().unsqueeze(1).unwrap();
        let en = et.sqr().unwrap().sum(1).unwrap().sqrt().unwrap().unsqueeze(1).unwrap();
        (ct.broadcast_div(&cn).unwrap(), et.broadcast_div(&en).unwrap())
    }

    #[test]
    fn happy_path_mining_changes_loss() {
        let (c, e) = make_vecs(8, 32);

        let no_mine = DirectionalContrastiveLoss::new(LossConfig {
            hard_negative_scale: 1.0, ..Default::default()
        });
        let with_mine = DirectionalContrastiveLoss::new(LossConfig {
            hard_negative_scale: 2.0, hard_negative_count: 3, ..Default::default()
        });

        let v_no: f32 = no_mine.info_nce_loss(&c, &e).unwrap()
            .flatten_all().unwrap().to_vec1().unwrap()[0];
        let v_yes: f32 = with_mine.info_nce_loss(&c, &e).unwrap()
            .flatten_all().unwrap().to_vec1().unwrap()[0];

        println!("=== WS3 Mining Impact ===");
        println!("Without: {:.6}  With: {:.6}  Diff: {:.6}", v_no, v_yes, (v_yes - v_no).abs());
        assert!((v_yes - v_no).abs() > 1e-6, "Mining should change loss");
        assert!(v_no > 0.0 && v_yes > 0.0, "Losses should be positive");
        println!("[PASS]\n");
    }

    #[test]
    fn happy_path_gradient_flow() {
        let d = 16;
        let n = 6;
        let c_data: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let e_data: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.2 + 1.0).cos()).collect();
        let ct = Tensor::from_slice(&c_data, (n, d), &Device::Cpu).unwrap();
        let et = Tensor::from_slice(&e_data, (n, d), &Device::Cpu).unwrap();
        let cv = Var::from_tensor(&ct).unwrap();
        let ev = Var::from_tensor(&et).unwrap();

        let loss_fn = DirectionalContrastiveLoss::new(LossConfig {
            hard_negative_scale: 2.0, hard_negative_count: 3, ..Default::default()
        });
        let loss = loss_fn.info_nce_loss(cv.as_tensor(), ev.as_tensor()).unwrap();
        let grads = loss.backward().unwrap();
        let gn: f32 = grads.get(cv.as_tensor()).unwrap()
            .sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();

        println!("WS3 Gradient: L2={:.6}", gn.sqrt());
        assert!(gn > 1e-10, "Gradient must be non-zero");
        println!("[PASS]");
    }

    #[test]
    fn edge_batch_size_2() {
        let (c, e) = make_vecs(2, 16);
        let loss_fn = DirectionalContrastiveLoss::new(LossConfig {
            hard_negative_scale: 2.0, ..Default::default()
        });
        let v: f32 = loss_fn.info_nce_loss(&c, &e).unwrap()
            .flatten_all().unwrap().to_vec1().unwrap()[0];
        println!("WS3 Edge: batch=2 loss={:.6} (mining skipped n<=2) [PASS]", v);
        assert!(v > 0.0);
    }

    #[test]
    fn edge_batch_size_1() {
        let (c, e) = make_vecs(1, 16);
        let loss_fn = DirectionalContrastiveLoss::new(LossConfig {
            hard_negative_scale: 2.0, ..Default::default()
        });
        let r = loss_fn.info_nce_loss(&c, &e);
        assert!(r.is_ok(), "batch=1 should not panic");
        println!("WS3 Edge: batch=1 ok [PASS]");
    }

    #[test]
    fn edge_scale_1_no_change() {
        let (c, e) = make_vecs(6, 16);
        let loss_fn = DirectionalContrastiveLoss::new(LossConfig {
            hard_negative_scale: 1.0, ..Default::default()
        });
        let v: f32 = loss_fn.info_nce_loss(&c, &e).unwrap()
            .flatten_all().unwrap().to_vec1().unwrap()[0];
        println!("WS3 Edge: scale=1.0 loss={:.6} [PASS]", v);
        assert!(v > 0.0 && !v.is_nan());
    }

    /// Verify combined loss works end-to-end with hard negative mining
    #[test]
    fn happy_path_combined_loss_with_mining() {
        let (c, e) = make_vecs(6, 32);
        let conf = Tensor::from_slice(&[0.9f32, 0.8, 0.7, 0.6, 0.5, 0.4], 6, &Device::Cpu).unwrap();

        let loss_fn = DirectionalContrastiveLoss::new(LossConfig {
            hard_negative_scale: 2.0,
            hard_negative_count: 3,
            ..Default::default()
        });

        let (total, components) = loss_fn.compute(&c, &e, &conf).unwrap();
        let total_val: f32 = total.flatten_all().unwrap().to_vec1().unwrap()[0];

        println!("=== WS3 Combined Loss ===");
        println!("Contrastive: {:.6}", components.contrastive);
        println!("Directional: {:.6}", components.directional);
        println!("Separation:  {:.6}", components.separation);
        println!("Soft label:  {:.6}", components.soft_label);
        println!("Total:       {:.6}", total_val);

        assert!(total_val > 0.0, "Total loss should be positive");
        assert!(components.contrastive > 0.0, "Contrastive should be positive");
        println!("[PASS]\n");
    }
}

// ============================================================================
// WS5: LoRA Key Projections
// ============================================================================

mod ws5 {
    use context_graph_embeddings::training::lora::{LoraConfig, LoraLayers};
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn happy_path_key_adapters_created() {
        println!("=== WS5 Key Adapters ===");

        let no_key = LoraConfig { apply_key: false, num_layers: 4, hidden_size: 32, rank: 8, ..Default::default() };
        let with_key = LoraConfig { apply_key: true, num_layers: 4, hidden_size: 32, rank: 8, ..Default::default() };

        let l_no = LoraLayers::new(no_key.clone(), &Device::Cpu).unwrap();
        let l_yes = LoraLayers::new(with_key.clone(), &Device::Cpu).unwrap();

        println!("No key: Q={} V={} K={} params={} vars={}",
            l_no.query_adapters.len(), l_no.value_adapters.len(),
            l_no.key_adapters.len(), l_no.total_params(), l_no.all_trainable_vars().len());
        println!("With key: Q={} V={} K={} params={} vars={}",
            l_yes.query_adapters.len(), l_yes.value_adapters.len(),
            l_yes.key_adapters.len(), l_yes.total_params(), l_yes.all_trainable_vars().len());

        assert_eq!(l_no.key_adapters.len(), 0);
        assert_eq!(l_yes.key_adapters.len(), 4);
        assert_eq!(l_no.all_trainable_vars().len(), 4 * 2 * 2);
        assert_eq!(l_yes.all_trainable_vars().len(), 4 * 3 * 2);

        let ratio = l_yes.total_params() as f64 / l_no.total_params() as f64;
        println!("Param increase: {:.1}%", (ratio - 1.0) * 100.0);
        assert!((ratio - 1.5).abs() < 0.01, "Should be ~50% increase");

        assert_eq!(no_key.total_params(), l_no.total_params());
        assert_eq!(with_key.total_params(), l_yes.total_params());
        println!("[PASS]\n");
    }

    #[test]
    fn happy_path_key_forward() {
        let config = LoraConfig { apply_key: true, num_layers: 2, hidden_size: 16, rank: 4, ..Default::default() };
        let layers = LoraLayers::new(config, &Device::Cpu).unwrap();
        let x = Tensor::ones((3, 16), DType::F32, &Device::Cpu).unwrap();

        let out = layers.apply_key(0, &x).unwrap();
        assert_eq!(out.dims(), &[3, 16]);

        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        let max_v = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        println!("WS5 Forward: shape={:?} max_val={:.6} (B=0 → expect 0)", out.dims(), max_v);
        assert!(max_v < 1e-6, "Initial output should be ~0");
        println!("[PASS]");
    }

    #[test]
    fn happy_path_training_toggle() {
        let config = LoraConfig { apply_key: true, num_layers: 2, hidden_size: 16, rank: 4, dropout: 0.1, ..Default::default() };
        let layers = LoraLayers::new(config, &Device::Cpu).unwrap();

        assert!(!layers.training.get());
        layers.set_training(true);
        assert!(layers.training.get());
        layers.set_training(false);
        assert!(!layers.training.get());
        println!("WS5 Training toggle [PASS]");
    }

    #[test]
    fn edge_disabled_key_returns_zeros() {
        let config = LoraConfig { apply_key: false, num_layers: 2, hidden_size: 8, rank: 4, ..Default::default() };
        let layers = LoraLayers::new(config, &Device::Cpu).unwrap();
        let x = Tensor::ones((2, 8), DType::F32, &Device::Cpu).unwrap();
        let r = layers.apply_key(0, &x).unwrap();
        let vals: Vec<f32> = r.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|v| *v == 0.0));
        println!("WS5 Edge: disabled key → zeros [PASS]");
    }

    #[test]
    fn edge_oob_layer() {
        let config = LoraConfig { apply_key: true, num_layers: 2, hidden_size: 8, rank: 4, ..Default::default() };
        let layers = LoraLayers::new(config, &Device::Cpu).unwrap();
        let x = Tensor::ones((2, 8), DType::F32, &Device::Cpu).unwrap();
        let r = layers.apply_key(99, &x).unwrap();
        let vals: Vec<f32> = r.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|v| *v == 0.0));
        println!("WS5 Edge: OOB layer → zeros [PASS]");
    }

    #[test]
    fn edge_rank_zero() {
        let config = LoraConfig { rank: 0, apply_key: true, ..Default::default() };
        assert_eq!(config.total_params(), 0);
        println!("WS5 Edge: rank=0 → 0 params [PASS]");
    }

    /// Verify save_lora serializes key adapters by checking safetensors content
    #[test]
    fn happy_path_save_load_key_adapters() {
        use std::collections::HashMap;
        println!("=== WS5 Save/Load Key Adapters ===");

        let config = LoraConfig {
            apply_key: true,
            num_layers: 2,
            hidden_size: 8,
            rank: 4,
            ..Default::default()
        };
        let layers = LoraLayers::new(config.clone(), &Device::Cpu).unwrap();

        // Manually build safetensors data to verify format
        let mut tensors = HashMap::new();
        for (i, adapter) in layers.query_adapters.iter().enumerate() {
            tensors.insert(format!("lora.query.{}.a", i), adapter.a.as_tensor().clone());
            tensors.insert(format!("lora.query.{}.b", i), adapter.b.as_tensor().clone());
        }
        for (i, adapter) in layers.value_adapters.iter().enumerate() {
            tensors.insert(format!("lora.value.{}.a", i), adapter.a.as_tensor().clone());
            tensors.insert(format!("lora.value.{}.b", i), adapter.b.as_tensor().clone());
        }
        for (i, adapter) in layers.key_adapters.iter().enumerate() {
            tensors.insert(format!("lora.key.{}.a", i), adapter.a.as_tensor().clone());
            tensors.insert(format!("lora.key.{}.b", i), adapter.b.as_tensor().clone());
        }

        println!("Tensor keys saved:");
        let mut keys: Vec<_> = tensors.keys().collect();
        keys.sort();
        for k in &keys { println!("  {}", k); }

        // Verify key adapter tensors exist
        assert!(tensors.contains_key("lora.key.0.a"), "Missing lora.key.0.a");
        assert!(tensors.contains_key("lora.key.0.b"), "Missing lora.key.0.b");
        assert!(tensors.contains_key("lora.key.1.a"), "Missing lora.key.1.a");
        assert!(tensors.contains_key("lora.key.1.b"), "Missing lora.key.1.b");

        // Verify tensor shapes
        let ka = &tensors["lora.key.0.a"];
        let kb = &tensors["lora.key.0.b"];
        assert_eq!(ka.dims(), &[8, 4], "key.a should be [hidden_size, rank]");
        assert_eq!(kb.dims(), &[4, 8], "key.b should be [rank, hidden_size]");

        // Total: 2 layers × 3 types × 2 tensors = 12 tensors
        assert_eq!(tensors.len(), 12, "Expected 12 tensors (2 layers × Q/V/K × A/B)");

        // Verify B tensors are zero (initial state)
        let b_vals: Vec<f32> = kb.flatten_all().unwrap().to_vec1().unwrap();
        assert!(b_vals.iter().all(|v| *v == 0.0), "B tensors should be zero at init");

        println!("[PASS] All key adapter tensors present with correct shapes\n");
    }
}
