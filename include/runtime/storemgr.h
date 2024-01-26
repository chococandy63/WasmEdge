// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2022 Second State INC

//===-- wasmedge/runtime/storemgr.h - Store Manager definition ------------===//
//
// Part of the WasmEdge Project.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of Store Manager.
///
//===----------------------------------------------------------------------===//
#pragma once

#include "runtime/instance/module.h"

#include <list>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace WasmEdge {

namespace Executor {
class Executor;
} // namespace Executor

namespace Runtime {

class Environment {
  std::shared_ptr<Environment> Parent;
  std::map<std::string, const Instance::ModuleInstance *, std::less<>> NamedMod;

  friend class Executor::Executor;

public:
  Environment() {}
  Environment(std::shared_ptr<Environment> P) : Parent{P} {}

  /// Register named module into this store.
  Expect<void> registerModule(std::string_view Name,
                              const Instance::ModuleInstance *ModInst) {
    auto Iter = NamedMod.find(Name);
    if (likely(Iter != NamedMod.cend())) {
      return Unexpect(ErrCode::Value::ModuleNameConflict);
    }
    NamedMod.emplace(ModInst->getModuleName(), ModInst);
    return {};
  }

  /// Find module by name.
  const Instance::ModuleInstance *findModule(std::string_view Name) const {
    auto Iter = NamedMod.find(Name);
    if (likely(Iter != NamedMod.cend())) {
      return Iter->second;
    }
    if (Parent) {
      return Parent->findModule(Name);
    }
    return nullptr;
  }

  /// Reset this store manager and unlink all the registered module instances.
  void reset(StoreManager *StoreMgr) noexcept {
    for (auto &&Pair : NamedMod) {
      (const_cast<Instance::ModuleInstance *>(Pair.second))
          ->unlinkStore(StoreMgr);
    }
    NamedMod.clear();
    if (Parent) {
      Parent->reset(StoreMgr);
    }
  }

  void removeModule(std::string_view Name) {
    NamedMod.erase(std::string(Name));
  }

  /// Get the length of the list of registered modules.
  uint32_t getModuleListSize() const noexcept {
    return static_cast<uint32_t>(NamedMod.size());
  }

  /// Get list of registered modules.
  template <typename CallbackT> auto getModuleList(CallbackT &&CallBack) const {
    return std::forward<CallbackT>(CallBack)(NamedMod);
  }

  std::shared_ptr<Environment> getParent() const noexcept { return Parent; }
};

class StoreManager {
public:
  StoreManager() = default;
  ~StoreManager() {
    // When destroying this store manager, unlink all the registered module
    // instances.
    reset();
  }

  const Instance::ModuleInstance *getModuleByDBI(uint32_t Index) const {
    std::shared_lock Lock(Mutex);
    auto Iter = InstList.begin();
    for (uint32_t I = Index; I > 0; --I) {
      Iter++;
    }
    return *Iter;
  }

  /// Get the length of the list of registered modules.
  uint32_t getModuleListSize() const noexcept {
    std::shared_lock Lock(Mutex);
    return Env.getModuleListSize();
  }
  /// Get list of registered modules.
  template <typename CallbackT> auto getModuleList(CallbackT &&CallBack) const {
    std::shared_lock Lock(Mutex);
    return Env.getModuleList(CallBack);
  }
  const Instance::ModuleInstance *findModule(std::string_view Name) const {
    std::shared_lock Lock(Mutex);
    return Env.findModule(Name);
  }
  /// Reset this store manager and unlink all the registered module instances.
  void reset() noexcept {
    std::shared_lock Lock(Mutex);
    return Env.reset(this);
  }

private:
  /// \name Mutex for thread-safe.
  mutable std::shared_mutex Mutex;

  friend class Executor::Executor;

  /// Register named module into this store.
  Expect<void> registerModule(const Instance::ModuleInstance *ModInst) {
    return registerModule(ModInst->getModuleName(), ModInst);
  }
  Expect<void> registerModule(std::string_view Name,
                              const Instance::ModuleInstance *ModInst) {
    std::shared_lock Lock(Mutex);
    if (auto Res = Env.registerModule(Name, ModInst)) {
      InstList.push_front(ModInst);
      // Link the module instance to this store manager.
      (const_cast<Instance::ModuleInstance *>(ModInst))
          ->linkStore(this, [](StoreManager *Store,
                               const Instance::ModuleInstance *Inst) {
            // The unlink callback.
            std::unique_lock CallbackLock(Store->Mutex);
            (Store->Env).removeModule(std::string(Inst->getModuleName()));
          });
    } else {
      return Unexpect(Res);
    }
    return {};
  }

  /// Collect the instantiation failed module.
  void recycleModule(std::unique_ptr<Instance::ModuleInstance> &&Mod) {
    FailedMod = std::move(Mod);
  }

  void pushNamespace() noexcept {
    std::unique_lock Lock(Mutex);
    auto P = std::make_shared<Environment>(Env);
    Env = Environment(P);
  }

  void popNamespace() noexcept {
    std::unique_lock Lock(Mutex);
    Env = *Env.getParent();
  }

  /// \name Module instance order
  /// Create for De-Bruijn indicies
  std::list<const Instance::ModuleInstance *> InstList;
  /// \name Module name mapping.
  Environment Env;

  /// \name Last instantiation failed module.
  /// According to the current spec, the instances should be able to be
  /// referenced even if instantiation failed. Therefore store the failed module
  /// instance here to keep the instances.
  /// FIXME: Is this necessary to be a vector?
  std::unique_ptr<Instance::ModuleInstance> FailedMod;
};

} // namespace Runtime
} // namespace WasmEdge
