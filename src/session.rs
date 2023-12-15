use std::cell::RefCell;
use std::collections::HashMap;
use std::error::Error;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use uuid::Uuid;

use crate::error::TensoriaError;
use crate::var::{TensorData, Variable, VarType};

pub struct Session {
    pub(crate) tensors: Arc<Mutex<HashMap<Uuid, Arc<Variable>>>>,
}

impl Session {
    pub fn new() -> Self {
        Session {
            tensors: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn new_tensor_var(&self, data: Option<TensorData>, shape: Vec<usize>) -> Result<Arc<Variable>, Box<dyn Error>> {
        let is_shape_valid = match &data {
            Some(values) => {
                match (values.len(), shape.len()) {
                    (1, 0) | (1, 1) => { true }
                    _ => {
                        let len_from_shape = shape.iter().fold(1, |x, y| x * (*y as usize));
                        values.len() == len_from_shape
                    }
                }
            }
            None => true
        };
        if !is_shape_valid {
            return Err(Box::new(TensoriaError::CannotReshapeError {}));
        }

        let tensor = Arc::new(Variable {
            id: Uuid::new_v4(),
            tensor_data: data,
            shape,
            session: Arc::downgrade(&self.tensors),
            prevs: vec![],
            nexts: Rc::new(RefCell::new(Vec::new())),
            var_type: VarType::Leaf,
        });
        self.tensors.lock().unwrap().insert(tensor.id, tensor.clone());
        Ok(tensor)
    }

    pub fn dfs(&self, root: Uuid, out: &mut Vec<Uuid>) {
        let p_visited = out.contains(&root);
        if p_visited {
            return;
        }

        let var = &self.tensors.lock().unwrap()[&root].prevs.clone();
        for p in var {
            self.dfs(*p, out);
        }
        out.push(root);
    }

    pub fn sorted_ids(&self) -> Vec<Uuid> {
        let terminal_ids: Vec<Uuid> = self.tensors.lock().unwrap()
            .iter()
            .filter(|(_, v)| v.nexts.borrow().len() == 0)
            .map(|(_, v)| v.id)
            .collect();

        let mut out: Vec<Uuid> = vec![];
        for id in terminal_ids {
            self.dfs(id, &mut out);
        }
        out
    }
}


#[cfg(test)]
mod test {
    use crate::session::Session;

    #[test]
    fn topological_sort() {
        let sess = Session::new();
        let a = sess.new_tensor_var(None, vec![1]).unwrap();
        let b = sess.new_tensor_var(None, vec![2]).unwrap();
        let c = sess.new_tensor_var(None, vec![3]).unwrap();
        let d = sess.new_tensor_var(None, vec![4]).unwrap();

        let add_res = a.add(&b);
        let sub_res = add_res.sub(&c);
        let final_res = sub_res.add(&d);
        let sorted = sess.sorted_ids();

        assert_eq!(sorted, vec![a.id, b.id, add_res.id, c.id, sub_res.id, d.id, final_res.id]);
    }
}